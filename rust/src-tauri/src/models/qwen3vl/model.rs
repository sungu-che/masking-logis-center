use std::io::{Read, Seek};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_core::quantized::{QMatMul, gguf_file};
use candle_nn::{
    Activation, Embedding, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{
            eager_attention_forward, get_layer_norm,
            gguf::{Gguf, ProjKind, QuantizedLinear, TwoLinearMLPGguf},
        },
        qwen3::model::Qwen3DecoderLayer,
        qwen3vl::config::{
            Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig, qwen3vl_text_config2qwen3_config,
        },
    },
    position_embed::rope::{
        Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb_vision,
    },
    utils::tensor_utils::{
        bitor_tensor, get_vision_next_indices, linspace, mask_index_add, masked_scatter_dim0,
        nonzero_index, prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
        zero_index,
    },
};

/// 🌟 [VISION-STREAM] Gguf 래퍼는 Content 를 소유권으로 받는데 Content 는 Clone 이 아니라
///   Arc<Content> 로 공유할 수 없습니다. Qwen3_5TextModel::reload_layer 와 동일하게
///   &Content + &mut Reader 로 직접 읽는 경로를 둡니다.
///   이렇게 해야 레이어 재로드마다 GGUF 헤더/텐서인덱스를 재파싱하지 않습니다.
fn ct_dequant_f16<R: Read + Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    name: &str,
) -> Result<Tensor> {
    let t = ct.tensor(reader, name, device)?;
    Ok(t.dequantize_f16(device).or_else(|_| t.dequantize(device))?)
}

fn ct_layer_norm<R: Read + Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    prefix: &str,
    eps: f64,
) -> Result<LayerNorm> {
    let weight = ct_dequant_f16(ct, reader, device, &format!("{prefix}.weight"))?
        .to_dtype(DType::F32)?;
    match ct_dequant_f16(ct, reader, device, &format!("{prefix}.bias")) {
        Ok(b) => Ok(LayerNorm::new(weight, b.to_dtype(DType::F32)?, eps)),
        Err(_) => Ok(LayerNorm::new_no_bias(weight, eps)),
    }
}

fn ct_qlinear<R: Read + Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    prefix: &str,
    bias: bool,
) -> Result<QuantizedLinear> {
    let w = ct.tensor(reader, &format!("{prefix}.weight"), device)?;
    let qm = QMatMul::from_qtensor(w)?;
    let b = if bias {
        ct_dequant_f16(ct, reader, device, &format!("{prefix}.bias")).ok()
    } else {
        None
    };
    Ok(QuantizedLinear::new(qm, b))
}

pub struct Qwen3VLVisionPatchEmbed {
    conv3d_weight: Tensor,
    conv3d_bias: Tensor,
}

impl Qwen3VLVisionPatchEmbed {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let temporal_patch_size = cfg.temporal_patch_size;
        let in_channels = cfg.in_channels;
        let embed_dim = cfg.hidden_size;
        // conv3d weight key: visual.patch_embed.proj.weight, value: Tensor[dims 1024, 3, 2, 16, 16; bf16, cuda:0]
        // (1024, 3, 2, 16, 16) -> (1024, 1536) -> (1536, 1024)
        let conv3d_weight = vb
            .get_with_hints(
                (
                    embed_dim,
                    in_channels,
                    temporal_patch_size,
                    patch_size,
                    patch_size,
                ),
                "proj.weight",
                Init::Const(1.),
            )?
            .flatten(1, 4)?
            .t()?;
        // (1024) -> (1, 1024)
        let conv3d_bias = vb
            .get_with_hints((embed_dim,), "proj.bias", Init::Const(0.))?
            .unsqueeze(0)?;
        Ok(Self {
            conv3d_weight,
            conv3d_bias,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>) -> Result<Self> {
        
        let conv3d_weight_0 = gguf.get_dequantized_f16("v.patch_embd.weight")?.to_dtype(candle_core::DType::F16)?.unsqueeze(2)?;
        let conv3d_weight_1 = gguf.get_dequantized_f16("v.patch_embd.weight.1")?.to_dtype(candle_core::DType::F16)?.unsqueeze(2)?;
        let conv3d_weight = Tensor::cat(&[conv3d_weight_0, conv3d_weight_1], 2)?.flatten(1, 4)?.t()?;
        let conv3d_bias = gguf.get_dequantized_f16("v.patch_embd.bias")?.to_dtype(candle_core::DType::F16)?;
        Ok(Self {
            conv3d_weight,
            conv3d_bias,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states shape:  (grid_t*grid_h*grid_w, c*temporal_patch_size*patch_size*patch_size)
        // ((), 1536) matmul (1536, 1024) -> ((), 1024)
        let dtype = hidden_states.dtype();
        let hidden_states = hidden_states.matmul(&self.conv3d_weight.to_dtype(dtype)?)?;
        let hidden_states = hidden_states.broadcast_add(&self.conv3d_bias.to_dtype(dtype)?)?;
        Ok(hidden_states)
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.conv3d_weight = self.conv3d_weight.to_device(device)?;
        self.conv3d_bias = self.conv3d_bias.to_device(device)?;
        Ok(())
    }

    /// 🌟 [VISION-JIT] patch_embed 가중치 해제 (conv3d 전개 행렬은 단독으로도 수십 MB)
    pub fn clear_weights(&mut self) {
        let dummy = Tensor::zeros((1, 1), DType::F16, &Device::Cpu).unwrap();
        self.conv3d_weight = dummy.clone();
        self.conv3d_bias = dummy;
    }

    /// 🌟 [VISION-JIT] new_from_gguf와 100% 동일한 전개 순서로 in-place 재로드합니다.
    pub fn load_weights_inplace<R: Read + Seek>(&mut self, gguf: &mut Gguf<R>) -> Result<()> {
        let w0 = gguf
            .get_dequantized_f16("v.patch_embd.weight")?
            .to_dtype(candle_core::DType::F16)?
            .unsqueeze(2)?;
        let w1 = gguf
            .get_dequantized_f16("v.patch_embd.weight.1")?
            .to_dtype(candle_core::DType::F16)?
            .unsqueeze(2)?;
        self.conv3d_weight = Tensor::cat(&[w0, w1], 2)?.flatten(1, 4)?.t()?;
        self.conv3d_bias = gguf
            .get_dequantized_f16("v.patch_embd.bias")?
            .to_dtype(candle_core::DType::F16)?;
        Ok(())
    }
}

pub struct Qwen3VLVisionPatchMerger {
    hidden_size: usize,
    use_postshuffle_norm: bool,
    norm: LayerNorm,
    // linear_fc1: Linear,
    linear_fc1: ProjKind,
    act_fn: Activation,
    // linear_fc2: Linear,
    linear_fc2: ProjKind,
}

impl Qwen3VLVisionPatchMerger {
    pub fn new(
        config: &Qwen3VLVisionConfig,
        vb: VarBuilder,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size * config.spatial_merge_size.pow(2);
        let norm_size = if use_postshuffle_norm {
            hidden_size
        } else {
            config.hidden_size
        };
        let norm = get_layer_norm(vb.pp("norm"), 1e-6, norm_size, true)?;
        let linear_fc1 = linear(hidden_size, hidden_size, vb.pp("linear_fc1"))?;
        let act_fn = Activation::Gelu;
        let linear_fc2 = linear(hidden_size, config.out_hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            hidden_size,
            use_postshuffle_norm,
            norm,
            linear_fc1: ProjKind::LinearProj(linear_fc1),
            act_fn,
            linear_fc2: ProjKind::LinearProj(linear_fc2),
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        rms_norm_eps: f64,
        use_postshuffle_norm: bool,
        hidden_size: usize,
        spatial_merge_size: usize,
        norm_prefix: &str,
        linear1_prefix: &str,
        linear2_prefix: &str,
    ) -> Result<Self> {
        let hidden_size = hidden_size * spatial_merge_size.pow(2);
        let norm = gguf.layer_norm(norm_prefix, rms_norm_eps)?;
        let linear_1 = gguf.quantize_linear(linear1_prefix, true)?;
        let act_fn = Activation::Gelu;
        let linear_2 = gguf.quantize_linear(linear2_prefix, true)?;
        Ok(Self {
            hidden_size,
            use_postshuffle_norm,
            norm,
            linear_fc1: ProjKind::QuantizedProj(linear_1),
            act_fn,
            linear_fc2: ProjKind::QuantizedProj(linear_2),
        })
    }

    // 👇 GPU VRAM으로 텐서를 넘겨주기 위해 필수적인 누락되었던 함수입니다. 👇
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let n_w = self.norm.weight().to_device(device)?;
        let n_b = self.norm.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm = LayerNorm::new(n_w, n_b, 1e-6);
        Ok(())
    }

    /// 🌟 [VISION-JIT] merger(norm + fc1 + fc2) 가중치 전체 해제
    pub fn clear_weights(&mut self) {
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.norm = LayerNorm::new(dummy_t.clone(), dummy_t, 1e-6);
        let dummy_p = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.linear_fc1 = dummy_p.clone();
        self.linear_fc2 = dummy_p;
    }

    /// 🌟 [VISION-JIT] new_from_gguf와 동일한 prefix 규약으로 in-place 재로드합니다.
    pub fn load_weights_inplace<R: Read + Seek>(
        &mut self,
        gguf: &mut Gguf<R>,
        rms_norm_eps: f64,
        norm_prefix: &str,
        linear1_prefix: &str,
        linear2_prefix: &str,
    ) -> Result<()> {
        self.norm = gguf.layer_norm(norm_prefix, rms_norm_eps)?;
        self.linear_fc1 = ProjKind::QuantizedProj(gguf.quantize_linear(linear1_prefix, true)?);
        self.linear_fc2 = ProjKind::QuantizedProj(gguf.quantize_linear(linear2_prefix, true)?);
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.use_postshuffle_norm {
            xs.reshape(((), self.hidden_size))?
        } else {
            xs.clone()
        };
        let orig_dtype = xs.dtype();
        let xs = self
            .norm
            .forward(&xs.to_dtype(self.norm.weight().dtype())?)?
            .reshape(((), self.hidden_size))?;
        let xs = xs.to_dtype(orig_dtype)?;
        let xs = self
            .linear_fc2
            .forward(&self.act_fn.forward(&self.linear_fc1.forward(&xs)?)?)?;
        Ok(xs)
    }
}

pub struct Qwen3VLVisionAttention {
    num_heads: usize,
    // qkv: Linear,
    // proj: Linear,
    qkv: ProjKind,
    proj: ProjKind,
    scaling: f64,
}

impl Qwen3VLVisionAttention {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hidden_size / num_heads;
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            num_heads,
            qkv: ProjKind::LinearProj(qkv),
            proj: ProjKind::LinearProj(proj),
            scaling,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(mmproj_gguf: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let num_heads = mmproj_gguf
            .get_matedata("clip.vision.attention.head_count")?
            .to_u32()? as usize;
        let hidden_size = mmproj_gguf
            .get_matedata("clip.vision.embedding_length")?
            .to_u32()? as usize;
        let head_dim = hidden_size / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = mmproj_gguf.quantize_linear(&format!("{prefix}.attn_qkv"), true)?;
        let proj = mmproj_gguf.quantize_linear(&format!("{prefix}.attn_out"), true)?;
        Ok(Self {
            num_heads,
            qkv: ProjKind::QuantizedProj(qkv),
            proj: ProjKind::QuantizedProj(proj),
            scaling,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        chunks: &[usize], 
    ) -> Result<Tensor> {
        let seq_length = xs.dim(0)?;
        let qkv_states = self.qkv.forward(xs)?.reshape((seq_length, 3, self.num_heads, ()))?.permute((1, 0, 2, 3))?; 
        
        
        let query_states = qkv_states.i(0)?.contiguous()?; 
        let key_states = qkv_states.i(1)?.contiguous()?; 
        let value_states = qkv_states.i(2)?.contiguous()?; 
        
        let (query_states, key_states) = apply_rotary_pos_emb_vision(&query_states, &key_states, cos, sin)?;
        let query_states = query_states.transpose(0, 1)?.unsqueeze(0)?;
        let key_states = key_states.transpose(0, 1)?.unsqueeze(0)?;
        let value_states = value_states.transpose(0, 1)?.unsqueeze(0)?;
        
        let q_splits = split_tensor(&query_states, chunks, 2)?;
        let k_splits = split_tensor(&key_states, chunks, 2)?;
        let v_splits = split_tensor(&value_states, chunks, 2)?;

        // 🌟 [VISION-TILE] chunks 는 이미지당 h*w 이므로 이미지 1장이면 chunk 도 1개입니다.
        //    즉 q_len == kv_len == N 이 되어 전이 버퍼가 N × 4096 × heads 로 폭발합니다.
        //    쿼리 행은 소프트맥스 정규화 축(KV)과 무관하게 서로 독립이므로,
        //    쿼리를 타일로 잘라 계산한 뒤 이어 붙여도 결과가 완전히 동일합니다.
        let max_kv = chunks.iter().copied().max().unwrap_or(seq_length);
        let dtype_bytes = query_states.dtype().size_in_bytes();
        let q_tile = crate::models::common::vision_query_tile_size(self.num_heads, max_kv, dtype_bytes);

        let mut attn_outputs = Vec::new();
        for (q, (k, v)) in q_splits.iter().zip(k_splits.iter().zip(v_splits.iter())) {
            let q_len = q.dim(2)?;

            if q_tile == 0 || q_len <= q_tile {
                let output = eager_attention_forward(q, k, v, None, None, self.scaling)?;
                attn_outputs.push(output);
            } else {
                // eager_attention_forward 는 (b, q_l, h, d) 로 transpose 해서 돌려주므로
                // 타일 결과들을 dim 1(seq) 기준으로 순서대로 쌓으면 그대로 원본 순서가 됩니다.
                let mut off = 0usize;
                while off < q_len {
                    let take = (q_len - off).min(q_tile);
                    let q_tile_t = q.narrow(2, off, take)?.contiguous()?;
                    let out = eager_attention_forward(&q_tile_t, k, v, None, None, self.scaling)?;

                    // 🌟 [VRAM] 타일 입력은 결과 확보 즉시 폐기해 다음 타일이 그 자리를 쓰게 합니다.
                    drop(q_tile_t);

                    attn_outputs.push(out);
                    off += take;
                }
            }
        }
        
        let attn_output = Tensor::cat(&attn_outputs, 1)?.reshape((seq_length, ()))?;

        // 🌟 [VRAM] 조각들은 합쳐진 뒤 불필요합니다.
        drop(attn_outputs);

        Ok(self.proj.forward(&attn_output)?)
    }

    pub fn to_device(&mut self, _device: &Device) -> Result<()> {
        // GGUF 양자화된 텐서(ProjKind)는 자체 장치를 사용하므로 이동 생략
        Ok(())
    }

    /// 🌟 [VISION-JIT] qkv/proj 양자화 가중치 해제
    pub fn clear_weights(&mut self) {
        let dummy = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.qkv = dummy.clone();
        self.proj = dummy;
    }

    /// 🌟 [VISION-JIT] mmproj GGUF에서 qkv/proj를 in-place 재로드합니다.
    pub fn load_weights_inplace<R: Read + Seek>(
        &mut self,
        gguf: &mut Gguf<R>,
        prefix: &str,
    ) -> Result<()> {
        self.qkv = ProjKind::QuantizedProj(gguf.quantize_linear(&format!("{prefix}.attn_qkv"), true)?);
        self.proj = ProjKind::QuantizedProj(gguf.quantize_linear(&format!("{prefix}.attn_out"), true)?);
        Ok(())
    }
}

pub struct Qwen3VLVisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Qwen3VLVisionAttention,
    // mlp: TwoLinearMLP,
    mlp: TwoLinearMLPGguf,
}

impl Qwen3VLVisionBlock {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-6, config.hidden_size, true)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-6, config.hidden_size, true)?;
        let attn = Qwen3VLVisionAttention::new(config.clone(), vb.pp("attn"))?;
        // let mlp = TwoLinearMLP::new(
        //     vb.pp("mlp"),
        //     config.hidden_size,
        //     config.intermediate_size,
        //     config.hidden_size,
        //     config.hidden_act,
        //     true,
        //     "linear_fc1",
        //     "linear_fc2",
        // )?;
        let mlp = TwoLinearMLPGguf::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_size,
            config.hidden_act,
            true,
            "linear_fc1",
            "linear_fc2",
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        mmproj_gguf: &mut Gguf<R>,
        prefix: &str,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let norm1 = mmproj_gguf.layer_norm(&format!("{prefix}.ln1"), rms_norm_eps)?;
        let norm2 = mmproj_gguf.layer_norm(&format!("{prefix}.ln2"), rms_norm_eps)?;
        let attn = Qwen3VLVisionAttention::new_from_gguf(mmproj_gguf, prefix)?;
        let mlp = TwoLinearMLPGguf::new_from_gguf(
            mmproj_gguf,
            prefix,
            true,
            Some("ffn_up"),
            Some("ffn_down"),
            Some(Activation::GeluPytorchTanh),
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        chunks: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let orig_dtype = xs.dtype();

        let normed = self.norm1.forward(&xs.to_dtype(self.norm1.weight().dtype())?)?;
        let normed = normed.to_dtype(orig_dtype)?;
        let attn_out = self.attn.forward(&normed, cos, sin, chunks)?;
        let xs = xs.add(&attn_out)?;
        drop(attn_out);

        let normed2 = self.norm2.forward(&xs.to_dtype(self.norm2.weight().dtype())?)?;
        let normed2 = normed2.to_dtype(orig_dtype)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        let xs = xs.add(&mlp_out)?;
        drop(mlp_out);
        drop(normed2);

        Ok(xs)
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let n1_w = self.norm1.weight().to_device(device)?;
        let n1_b = self.norm1.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm1 = LayerNorm::new(n1_w, n1_b, 1e-6);

        let n2_w = self.norm2.weight().to_device(device)?;
        let n2_b = self.norm2.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm2 = LayerNorm::new(n2_w, n2_b, 1e-6);

        self.attn.to_device(device)?;
        Ok(())
    }

    /// 🌟 [VISION-JIT] 블록 단위 전체 해제 (norm1/norm2/attn/mlp)
    pub fn clear_weights(&mut self) {
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.norm1 = LayerNorm::new(dummy_t.clone(), dummy_t.clone(), 1e-6);
        self.norm2 = LayerNorm::new(dummy_t.clone(), dummy_t, 1e-6);
        self.attn.clear_weights();
        self.mlp.clear_weights();
    }

    /// 🌟 [VISION-JIT] new_from_gguf와 동일한 텐서 이름으로 블록 전체를 in-place 재로드합니다.
    pub fn load_weights_inplace<R: Read + Seek>(
        &mut self,
        gguf: &mut Gguf<R>,
        prefix: &str,
        rms_norm_eps: f64,
    ) -> Result<()> {
        self.norm1 = gguf.layer_norm(&format!("{prefix}.ln1"), rms_norm_eps)?;
        self.norm2 = gguf.layer_norm(&format!("{prefix}.ln2"), rms_norm_eps)?;
        self.attn.load_weights_inplace(gguf, prefix)?;
        self.mlp
            .load_weights_inplace(gguf, prefix, true, Some("ffn_up"), Some("ffn_down"))?;
        Ok(())
    }
}

pub struct Qwen3VLVisionModel {
    spatial_merge_size: usize,
    patch_embed: Qwen3VLVisionPatchEmbed,
    pos_embed: Embedding,
    num_grid_per_side: u32,
    rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
    blocks: Vec<Qwen3VLVisionBlock>,
    merger: Qwen3VLVisionPatchMerger,
    // 🌟 [VISION-CACHE] 캐시본의 deepstack 개수 검증을 위해 외부 노출이 필요합니다.
    pub deepstack_visual_indexes: Vec<usize>,
    deepstack_merger_list: Vec<Qwen3VLVisionPatchMerger>,
    dtype: DType,
    // 🌟 [VISION-JIT] 가중치 상주 여부. mmproj_mmap 이 None이면 재로드 소스가 없으므로 항상 상주로 고정됩니다.
    pub is_weights_loaded: bool,
    mmproj_path: Option<String>,
    // 🌟 [VISION-STREAM] 재로드마다 GGUF 헤더/텐서인덱스를 재파싱하지 않도록 상주시킵니다.
    //   레이어 스트리밍은 이미지당 27회 재로드가 발생하므로 이 상주가 전제조건입니다.
    mmproj_mmap: Option<Arc<memmap2::Mmap>>,
    mmproj_ct: Option<Arc<gguf_file::Content>>,
    // 🌟 [VISION-STREAM] 블록 단위 스트리밍 활성 여부.
    //   false 면 기존처럼 27개 블록이 통째로 상주합니다.
    pub stream_blocks: bool,
    rms_norm_eps: f64,
    hidden_size: usize,
    device: Device,
}

impl Qwen3VLVisionModel {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = config.spatial_merge_size;
        let patch_embed = Qwen3VLVisionPatchEmbed::new(&config, vb.pp("patch_embed"))?;
        let pos_embed = embedding(
            config.num_position_embeddings,
            config.hidden_size,
            vb.pp("pos_embed"),
        )?;
        let num_grid_per_side = (config.num_position_embeddings as f32).sqrt() as u32;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..config.depth {
            let block = Qwen3VLVisionBlock::new(config.clone(), vb_blocks.pp(i))?;
            blocks.push(block);
        }
        let merger = Qwen3VLVisionPatchMerger::new(&config, vb.pp("merger"), false)?;
        let deepstack_visual_indexes = config.deepstack_visual_indexes.clone();
        let mut deepstack_merger_list = Vec::new();
        let vb_deepstack = vb.pp("deepstack_merger_list");
        for i in 0..deepstack_visual_indexes.len() {
            let merger_i = Qwen3VLVisionPatchMerger::new(&config, vb_deepstack.pp(i), true)?;
            deepstack_merger_list.push(merger_i);
        }
        Ok(Self {
            spatial_merge_size,
            patch_embed,
            pos_embed,
            num_grid_per_side,
            rotary_pos_emb,
            blocks,
            merger,
            deepstack_visual_indexes,
            deepstack_merger_list,
            dtype: vb.dtype(),
            // 🌟 [VISION-JIT] safetensors(VarBuilder) 경로는 mmap 재로드 소스가 없으므로 JIT 비활성
            is_weights_loaded: true,
            mmproj_path: None,
            mmproj_mmap: None,
            mmproj_ct: None,
            stream_blocks: false,
            rms_norm_eps: 1e-6,
            hidden_size: config.hidden_size,
            device: vb.device().clone(),
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(mmproj_gguf: &mut Gguf<R>) -> Result<Self> {
        let spatial_merge_size = mmproj_gguf
            .get_matedata("clip.vision.spatial_merge_size")?
            .to_u32()? as usize;
        let patch_embed = Qwen3VLVisionPatchEmbed::new_from_gguf(mmproj_gguf)?;
        
        
        let pos_emb_weight = mmproj_gguf.get_dequantized_f16("v.position_embd.weight")?.to_dtype(candle_core::DType::F16)?;
        
        let hidden_size = mmproj_gguf
            .get_matedata("clip.vision.embedding_length")?
            .to_u32()? as usize;
        let pos_embed = Embedding::new(pos_emb_weight, hidden_size);
        let patch_size = mmproj_gguf
            .get_matedata("clip.vision.patch_size")?
            .to_u32()? as usize;
        let image_size = mmproj_gguf
            .get_matedata("clip.vision.image_size")?
            .to_u32()? as usize;
        let num_grid_per_side = image_size / patch_size;
        let num_heads = mmproj_gguf
            .get_matedata("clip.vision.attention.head_count")?
            .to_u32()? as usize;
        let head_dim = hidden_size / num_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let rms_norm_eps = mmproj_gguf
            .get_matedata("clip.vision.attention.layer_norm_epsilon")?
            .to_f32()? as f64;
        let mut blocks = Vec::new();
        let num_block = mmproj_gguf
            .get_matedata("clip.vision.block_count")?
            .to_u32()? as usize;
        for i in 0..num_block {
            let prefix = format!("v.blk.{i}");
            let block = Qwen3VLVisionBlock::new_from_gguf(mmproj_gguf, &prefix, rms_norm_eps)?;
            blocks.push(block);
        }
        let merger = Qwen3VLVisionPatchMerger::new_from_gguf(
            mmproj_gguf,
            rms_norm_eps,
            false,
            hidden_size,
            spatial_merge_size,
            "v.post_ln",
            "mm.0",
            "mm.2",
        )?;
        let mut deepstack_merger_list = Vec::new();
        let is_deepstack = mmproj_gguf
            .get_matedata("clip.vision.is_deepstack_layers")?
            .to_vec()?
            .iter()
            .map(|b| b.to_bool())
            .collect::<Result<Vec<bool>, candle_core::Error>>()?;
        let deepstack_visual_indexes = is_deepstack
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect::<Vec<usize>>();
        for i in &deepstack_visual_indexes {
            let prefix = format!("v.deepstack.{i}");
            let merger_i = Qwen3VLVisionPatchMerger::new_from_gguf(
                mmproj_gguf,
                rms_norm_eps,
                true,
                hidden_size,
                spatial_merge_size,
                &format!("{prefix}.norm"),
                &format!("{prefix}.fc1"),
                &format!("{prefix}.fc2"),
            )?;
            deepstack_merger_list.push(merger_i);
        }
        Ok(Self {
            spatial_merge_size,
            patch_embed,
            pos_embed,
            num_grid_per_side: num_grid_per_side as u32,
            rotary_pos_emb,
            blocks,
            merger,
            deepstack_visual_indexes,
            deepstack_merger_list,
            dtype: DType::F32,
            // 🌟 [VISION-JIT] 생성 직후에는 상주 상태. set_mmproj_source() 등록 후에만 unload가 허용됩니다.
            is_weights_loaded: true,
            mmproj_path: None,
            mmproj_mmap: None,
            mmproj_ct: None,
            stream_blocks: false,
            rms_norm_eps,
            hidden_size,
            device: mmproj_gguf.device().clone(),
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embed.to_device(device)?;
        let p_w = self.pos_embed.embeddings().to_device(device)?;
        self.pos_embed = Embedding::new(p_w, self.pos_embed.hidden_size());
        for block in self.blocks.iter_mut() { block.to_device(device)?; }
        self.merger.to_device(device)?;
        for merger in self.deepstack_merger_list.iter_mut() { merger.to_device(device)?; }
        Ok(())
    }

    /// 🌟 [VISION-JIT] mmproj GGUF 경로를 등록해야 unload/reload가 활성화됩니다.
    /// 등록하지 않으면 unload_weights()는 no-op으로 동작하여 기존 거동을 100% 보존합니다.
    ///
    /// 🌟 [VISION-STREAM] 경로만 받던 기존 방식은 재로드마다 Content::read 로
    ///   GGUF 텐서 인덱스 전체를 재파싱했습니다. mmap 과 Content 를 함께 상주시켜
    ///   재로드 비용을 "필요한 텐서만 읽기" 수준으로 낮춥니다.
    pub fn set_mmproj_source(&mut self, path: Option<String>) -> Result<()> {
        self.mmproj_path = path.clone();
        self.mmproj_mmap = None;
        self.mmproj_ct = None;

        let p = match path {
            Some(p) => p,
            None => return Ok(()),
        };

        let file = std::fs::File::open(&p)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mmap = Arc::new(mmap);

        let mut reader = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut reader)?;

        self.mmproj_mmap = Some(mmap);
        self.mmproj_ct = Some(Arc::new(ct));
        Ok(())
    }

    /// 하위 호환용 래퍼. 실패해도 조용히 JIT 만 비활성화됩니다.
    pub fn set_mmproj_path(&mut self, path: Option<String>) {
        if let Err(e) = self.set_mmproj_source(path) {
            println!("[VISION-JIT] mmproj source registration failed ({}). JIT disabled.", e);
            self.mmproj_mmap = None;
            self.mmproj_ct = None;
        }
    }

    /// 🌟 [VISION-STREAM] 블록 단위 스트리밍을 켜고 끕니다.
    ///   켜면 forward 가 블록 하나를 읽고 → 계산하고 → 즉시 버립니다.
    ///   가중치 상주는 337MB → 약 12MB 로 줄지만 이미지당 27회 PCIe 전송이 발생합니다.
    ///   쿼리축 타일링(Part A)으로 활성값 피크를 이미 잡았다면 대개 불필요합니다.
    pub fn set_block_streaming(&mut self, enabled: bool) {
        if enabled && self.mmproj_ct.is_none() {
            println!("[VISION-STREAM] mmproj source not registered. Streaming request ignored.");
            return;
        }
        self.stream_blocks = enabled;
        println!("[VISION-STREAM] Block streaming {}.", if enabled { "ENABLED" } else { "disabled" });
    }

    pub fn is_jit_capable(&self) -> bool {
        self.mmproj_ct.is_some() && self.mmproj_mmap.is_some()
    }

    /// 🌟 [VISION-JIT] ViT는 상태 없는 1회성 feed-forward이므로,
    /// 임베딩 주입이 끝나면 가중치를 1바이트 껍데기로 교체해 VRAM/RAM을 즉시 반환합니다.
    pub fn unload_weights(&mut self) {
        if !self.is_weights_loaded || self.mmproj_path.is_none() {
            return;
        }
        self.patch_embed.clear_weights();
        let dummy = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        self.pos_embed = Embedding::new(dummy, 1);
        for block in self.blocks.iter_mut() {
            block.clear_weights();
        }
        self.merger.clear_weights();
        for merger in self.deepstack_merger_list.iter_mut() {
            merger.clear_weights();
        }
        self.is_weights_loaded = false;

        if self.device.is_cuda() {
            let _ = self.device.synchronize();
        }
        Self::force_memory_release();
        println!("[VISION-JIT] mmproj weights unloaded. VRAM/RAM returned to OS.");
    }

    /// 🌟 [VISION-STREAM] 블록 i 하나만 mmap 에서 읽어 붙입니다.
    fn load_block(&mut self, i: usize) -> Result<()> {
        let (ct, mmap) = match (self.mmproj_ct.clone(), self.mmproj_mmap.clone()) {
            (Some(c), Some(m)) => (c, m),
            _ => return Ok(()),
        };
        let mut reader = std::io::Cursor::new(&mmap[..]);
        let prefix = format!("v.blk.{i}");
        let dev = self.device.clone();
        let eps = self.rms_norm_eps;

        self.blocks[i].norm1 = ct_layer_norm(&ct, &mut reader, &dev, &format!("{prefix}.ln1"), eps)?;
        self.blocks[i].norm2 = ct_layer_norm(&ct, &mut reader, &dev, &format!("{prefix}.ln2"), eps)?;
        self.blocks[i].attn.qkv =
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.attn_qkv"), true)?);
        self.blocks[i].attn.proj =
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.attn_out"), true)?);
        self.blocks[i].mlp.set_projs(
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.ffn_up"), true)?),
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.ffn_down"), true)?),
        );
        Ok(())
    }

    /// 🌟 [VISION-JIT] 블록을 제외한 공용 파트(patch_embed / pos_embed / merger 계열)만 붙입니다.
    fn load_shared_weights(&mut self) -> Result<()> {
        let (ct, mmap) = match (self.mmproj_ct.clone(), self.mmproj_mmap.clone()) {
            (Some(c), Some(m)) => (c, m),
            _ => return Ok(()),
        };
        let mut reader = std::io::Cursor::new(&mmap[..]);
        let dev = self.device.clone();
        let eps = self.rms_norm_eps;

        let w0 = ct_dequant_f16(&ct, &mut reader, &dev, "v.patch_embd.weight")?
            .to_dtype(DType::F16)?
            .unsqueeze(2)?;
        let w1 = ct_dequant_f16(&ct, &mut reader, &dev, "v.patch_embd.weight.1")?
            .to_dtype(DType::F16)?
            .unsqueeze(2)?;
        self.patch_embed.conv3d_weight = Tensor::cat(&[w0, w1], 2)?.flatten(1, 4)?.t()?;
        self.patch_embed.conv3d_bias =
            ct_dequant_f16(&ct, &mut reader, &dev, "v.patch_embd.bias")?.to_dtype(DType::F16)?;

        let pos_w = ct_dequant_f16(&ct, &mut reader, &dev, "v.position_embd.weight")?
            .to_dtype(DType::F16)?;
        self.pos_embed = Embedding::new(pos_w, self.hidden_size);

        self.merger.norm = ct_layer_norm(&ct, &mut reader, &dev, "v.post_ln", eps)?;
        self.merger.linear_fc1 =
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, "mm.0", true)?);
        self.merger.linear_fc2 =
            ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, "mm.2", true)?);

        let ds_indexes = self.deepstack_visual_indexes.clone();
        for (k, i) in ds_indexes.iter().enumerate() {
            let prefix = format!("v.deepstack.{i}");
            self.deepstack_merger_list[k].norm =
                ct_layer_norm(&ct, &mut reader, &dev, &format!("{prefix}.norm"), eps)?;
            self.deepstack_merger_list[k].linear_fc1 =
                ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.fc1"), true)?);
            self.deepstack_merger_list[k].linear_fc2 =
                ProjKind::QuantizedProj(ct_qlinear(&ct, &mut reader, &dev, &format!("{prefix}.fc2"), true)?);
        }
        Ok(())
    }

    /// 🌟 [VISION-JIT] 비전 입력이 감지되면 mmproj 에서 즉시 재로드합니다.
    /// stream_blocks 가 켜져 있으면 27개 블록은 건너뛰고 공용 파트만 붙입니다.
    /// (블록은 forward 안에서 하나씩 읽고 버립니다)
    pub fn reload_weights(&mut self) -> Result<()> {
        if self.is_weights_loaded {
            return Ok(());
        }
        if !self.is_jit_capable() {
            return Ok(());
        }

        self.load_shared_weights()?;

        if !self.stream_blocks {
            for i in 0..self.blocks.len() {
                self.load_block(i)?;
            }
        }

        self.is_weights_loaded = true;
        println!(
            "[VISION-JIT] mmproj weights reloaded (streaming: {}) from {}",
            self.stream_blocks,
            self.mmproj_path.as_deref().unwrap_or("<mmap>")
        );
        Ok(())
    }

    /// 🌟 [VISION-JIT] 기존 텍스트 모델과 동일한 OS 레벨 강제 메모리 반환 루틴
    fn force_memory_release() {
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
            let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        #[cfg(target_os = "linux")]
        unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
        #[cfg(target_os = "macos")]
        unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
    }

    pub fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let dev = grid_thw.device(); // 최종 목적지 (GPU)
        let cpu_dev = &Device::Cpu;  // 모든 연산은 CPU에서 수행하여 CUDA 에러 차단
        let grid_thw_cpu = grid_thw.to_device(cpu_dev)?.to_vec2::<u32>()?;
        
        // U32 커널 에러 차단을 위해 처음부터 F32로 연산
        let side_tensor = Tensor::new(self.num_grid_per_side as f32, cpu_dev)?;
        let one_t_f32 = Tensor::new(1f32, cpu_dev)?;
        
        let mut idx_tensors: [Vec<Tensor>; 4] = Default::default();
        let mut weight_tensors: [Vec<Tensor>; 4] = Default::default();
        let mut split_idx = vec![];

        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw_cpu[i][..] else { return Err(anyhow!("...")); };
            split_idx.push((h * w) as usize);
            let num_grid_per_side_sub_one = (self.num_grid_per_side - 1) as f32;
            let h_idxs = linspace(0.0, num_grid_per_side_sub_one, h as usize, cpu_dev)?;
            let w_idxs = linspace(0.0, num_grid_per_side_sub_one, w as usize, cpu_dev)?;
            
            let h_idxs_floor = h_idxs.floor()?;
            let w_idxs_floor = w_idxs.floor()?;
            
            let h_idxs_ceil = h_idxs_floor.broadcast_add(&one_t_f32)?.clamp(0f32, num_grid_per_side_sub_one)?;
            let w_idxs_ceil = w_idxs_floor.broadcast_add(&one_t_f32)?.clamp(0f32, num_grid_per_side_sub_one)?;
            
            let dh = h_idxs.sub(&h_idxs_floor)?.unsqueeze(D::Minus1)?;
            let dw = w_idxs.sub(&w_idxs_floor)?.unsqueeze(0)?;
            
            let base_h = h_idxs_floor.broadcast_mul(&side_tensor)?.unsqueeze(D::Minus1)?;
            let base_h_ceil = h_idxs_ceil.broadcast_mul(&side_tensor)?.unsqueeze(D::Minus1)?;

            // 안전한 CPU에서 U32 캐스팅
            idx_tensors[0].push(base_h.broadcast_add(&w_idxs_floor.unsqueeze(0)?)?.to_dtype(DType::U32)?.flatten_all()?);
            idx_tensors[1].push(base_h.broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?.to_dtype(DType::U32)?.flatten_all()?);
            idx_tensors[2].push(base_h_ceil.broadcast_add(&w_idxs_floor.unsqueeze(0)?)?.to_dtype(DType::U32)?.flatten_all()?);
            idx_tensors[3].push(base_h_ceil.broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?.to_dtype(DType::U32)?.flatten_all()?);

            let one_sub_dh = dh.affine(-1.0, 1.0)?;
            let one_sub_dw = dw.affine(-1.0, 1.0)?;

            weight_tensors[0].push(one_sub_dh.broadcast_mul(&one_sub_dw)?.flatten_all()?);
            weight_tensors[1].push(one_sub_dh.broadcast_mul(&dw)?.flatten_all()?);
            weight_tensors[2].push(dh.broadcast_mul(&one_sub_dw)?.flatten_all()?);
            weight_tensors[3].push(dh.broadcast_mul(&dw)?.flatten_all()?);
        }

        // 연산이 끝난 후 최종 결과만 GPU로 전송
        let idx_tensor = Tensor::stack(&[
            Tensor::cat(&idx_tensors[0], 0)?, Tensor::cat(&idx_tensors[1], 0)?,
            Tensor::cat(&idx_tensors[2], 0)?, Tensor::cat(&idx_tensors[3], 0)?,
        ], 0)?.to_device(dev)?; 

        let weight_tensor = Tensor::stack(&[
            Tensor::cat(&weight_tensors[0], 0)?, Tensor::cat(&weight_tensors[1], 0)?,
            Tensor::cat(&weight_tensors[2], 0)?, Tensor::cat(&weight_tensors[3], 0)?,
        ], 0)?.to_device(dev)?.to_dtype(self.dtype)?; 
        
        
        let pos_embeds = self.pos_embed.forward(&idx_tensor)?.to_dtype(self.dtype)?.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        
        let patch_pos_embeds = pos_embeds.i(0)?.add(&pos_embeds.i(1)?)?.add(&pos_embeds.i(2)?)?.add(&pos_embeds.i(3)?)?;
        
        let mut patch_pos_embeds_permute = vec![];
        let patch_pos_embeds = split_tensor(&patch_pos_embeds, &split_idx, 0)?;
        let merge_size = self.spatial_merge_size;
        for (i, pos_embed) in patch_pos_embeds.iter().enumerate() {
            let [t, h, w] = grid_thw_cpu[i][..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let pos_emebd_last_dim: usize = pos_embed.dim(D::Minus1)?;
            let pos_embed = pos_embed.repeat((t as usize, 1))?;
            let shape = Shape::from(vec![
                t as usize,
                h as usize / merge_size,
                merge_size,
                w as usize / merge_size,
                merge_size,
                pos_emebd_last_dim,
            ]);
            
            
            // 뒤따라오는 flatten(0, 4) 연산이 정상적으로 수행되며, 이미지가 깨지는 환각(공백 출력) 증세가 완벽하게 사라집니다!
            let pos_embed = pos_embed
                .reshape(shape)?
                .permute((0, 1, 3, 2, 4, 5))?
                .contiguous()? 
                .flatten(0, 4)?;
                
            patch_pos_embeds_permute.push(pos_embed);
        }
        
        let patch_pos_embeds = Tensor::cat(&patch_pos_embeds_permute, 0)?;
        Ok(patch_pos_embeds)
    }

    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let dev = grid_thw.device();
        let cpu_dev = &Device::Cpu;
        let merge_size = self.spatial_merge_size;
        let grid_thw_cpu = grid_thw.to_device(cpu_dev)?.to_vec2::<u32>()?;
        let max_hw = grid_thw_cpu.iter().flat_map(|thw| [thw[1], thw[2]]).max().unwrap_or(0);

        let freq_table = self.rotary_pos_emb.forward(max_hw as usize, dev)?; 
        let mut pos_ids_vec = vec![];
        
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw_cpu[i][..] else { return Err(anyhow!("...")); };
            let merged_h = h / merge_size as u32;
            let merged_w = w / merge_size as u32;
            
            // 모든 행렬 연산을 CPU에서 F32로 처리하여 CUDA 에러 방지
            let blocks_rows = Tensor::arange(0f32, merged_h as f32, cpu_dev)?;
            let blocks_cols = Tensor::arange(0f32, merged_w as f32, cpu_dev)?;
            let intra_row = Tensor::arange(0f32, merge_size as f32, cpu_dev)?;
            let intra_col = Tensor::arange(0f32, merge_size as f32, cpu_dev)?;

            let row_idx = blocks_rows
                .unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?
                .broadcast_mul(&Tensor::new(merge_size as f32, cpu_dev)?)?
                .broadcast_add(&intra_row.unsqueeze(0)?.unsqueeze(1)?.unsqueeze(3)?)?;

            let col_idx = blocks_cols
                .unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?
                .broadcast_mul(&Tensor::new(merge_size as f32, cpu_dev)?)?
                .broadcast_add(&intra_col.unsqueeze(0)?.unsqueeze(1)?.unsqueeze(2)?)?;

            let row_idx = row_idx.expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?.contiguous()?.flatten_all()?;
            let col_idx = col_idx.expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?.contiguous()?.flatten_all()?;
                
            let mut coords = Tensor::stack(&[row_idx, col_idx], D::Minus1)?.contiguous()?;
            if t > 1 { coords = coords.repeat((t as usize, 1))?; }
            
            // 연산 완료 후 최종적으로 GPU 이동 및 U32 변환
            pos_ids_vec.push(coords.to_device(dev)?.to_dtype(DType::U32)?);
        }
        let pos_ids = Tensor::cat(&pos_ids_vec, 0)?;
        let pos_ids_h = pos_ids.i((.., 0))?.contiguous()?; 
        let pos_ids_w = pos_ids.i((.., 1))?.contiguous()?; 
        
        let rotary_pos_emb_h = freq_table.index_select(&pos_ids_h, 0)?;
        let rotary_pos_emb_w = freq_table.index_select(&pos_ids_w, 0)?;
        
        Ok(Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?)
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let hidden_states = self.patch_embed.forward(hidden_states)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?.to_dtype(hidden_states.dtype())?;
        let hidden_states = hidden_states.broadcast_add(&pos_embeds)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(self.dtype)?;
        let sin = emb.sin()?.to_dtype(self.dtype)?;
        
        let grid_thw_cpu = grid_thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let mut chunks = Vec::new();
        for thw in grid_thw_cpu {
            let (t, h, w) = (thw[0] as usize, thw[1] as usize, thw[2] as usize);
            let hw = h * w;
            for _ in 0..t { chunks.push(hw); }
        }

        // 🌟 [VISION-STREAM] stream_blocks 가 꺼져 있으면 기존과 100% 동일한 경로입니다.
        //    켜져 있으면 블록 하나를 mmap 에서 읽고 → 계산하고 → 즉시 껍데기로 되돌립니다.
        //    상주 가중치가 27블록(약 337MB) → 1블록(약 12MB) 으로 줄지만,
        //    이미지당 27회 PCIe 전송이 추가됩니다.
        //    쿼리축 타일링으로 활성값 피크를 이미 잡았다면 대개 불필요한 트레이드오프입니다.
        let streaming = self.stream_blocks && self.is_jit_capable();
        let total_blocks = self.blocks.len();
        let ds_indexes = self.deepstack_visual_indexes.clone();

        let mut deepstack_feature_lists = vec![];
        for layer_num in 0..total_blocks {
            if streaming {
                self.load_block(layer_num)?;
            }

            hidden_states = self.blocks[layer_num].forward(&hidden_states, &chunks, &cos, &sin)?;

            if let Some(index) = ds_indexes.iter().position(|&x| x == layer_num) {
                let deepstack_feature = self.deepstack_merger_list[index].forward(&hidden_states)?;
                deepstack_feature_lists.push(deepstack_feature);
            }

            if streaming {
                // 계산이 끝난 블록은 즉시 반환합니다. 다음 블록이 그 자리를 씁니다.
                self.blocks[layer_num].clear_weights();
                if self.device.is_cuda() {
                    let _ = self.device.synchronize();
                }
            }
        }
        hidden_states = self.merger.forward(&hidden_states)?;

        if streaming {
            println!("[VISION-STREAM] {} blocks streamed. Resident block weights: ~1 block.", total_blocks);
        }

        Ok((hidden_states, deepstack_feature_lists))
    }
}

pub struct Qwen3VLTextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl Qwen3VLTextModel {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let embed_tokens = embedding(vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let qwen3_cfg = qwen3vl_text_config2qwen3_config(&config);
            let layer = Qwen3DecoderLayer::new(&qwen3_cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.mrope_section.clone();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    inputs_embeds.device(),
                )?)
            }
        };
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    xs = mask_index_add(
                        &xs.squeeze(0)?,
                        &visual_pos_masks.unwrap().squeeze(0)?,
                        &deepstack_embeds[layer_idx],
                    )?
                    .unsqueeze(0)?;
                }
            }
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn evacuate_kv_to_cpu(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.compress_kv_in_vram()?;
        }
        Ok(())
    }

    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers.iter().map(|l| l.get_kv_cache()).collect()
    }

    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, c) in self.layers.iter_mut().zip(cache.into_iter()) {
            layer.set_kv_cache(c);
        }
    }

    pub fn get_embed_tokens(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }
}

pub struct Qwen3VLModel {
    config: Qwen3VLConfig,
    visual: Qwen3VLVisionModel,
    language_model: Qwen3VLTextModel,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl Qwen3VLModel {
    pub fn new(config: Qwen3VLConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let config = config.clone();
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_m.pp("visual"))?;
        let language_model =
            Qwen3VLTextModel::new(config.text_config.clone(), vb_m.pp("language_model"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            config,
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
        })
    }

    // 🌟 [VISION-STREAM] Qwen3VLVisionModel::forward 가 블록을 하나씩 읽고 버리므로 &mut self 가 필요합니다.
    fn get_vision_features(
        &mut self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        // spatial_merge_size 는 스칼라이므로 借用 충돌을 피해 먼저 복사합니다.
        let merge_sq = self.visual.spatial_merge_size.pow(2);

        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(pixel_values, image_grid_thw)?;
        // torch.prod
        let split_sizes: Vec<usize> = prod_tensor_last_dim(image_grid_thw)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize / merge_sq)
            .collect();
        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
        Ok((image_embeds, deepstack_image_embeds))
    }

    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token_id = if is_image {
            self.config.image_token_id as u32
        } else {
            self.config.video_token_id as u32
        };
        
        
        let special_token = Tensor::new(vec![special_token_id], input_ids.device())?;
        let special_mask = input_ids
            .broadcast_eq(&special_token)?
            .to_dtype(candle_core::DType::U8)?;
            
        Ok(special_mask)
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let video_grid_thw = match video_grid_thw {
            Some(thw) => {
                let grid_t = thw.i((.., 0))?.to_vec1::<u32>()?;
                let mut v_thw_vec = Vec::new();
                for (index, t) in grid_t.iter().enumerate() {
                    let mut thw_i = thw.i(index)?.to_vec1::<u32>()?;
                    // [12, 30, 50]
                    // [1, 30, 50]*t
                    thw_i[0] = 1;
                    v_thw_vec.push(
                        Tensor::new(thw_i, thw.device())?
                            .repeat(*t as usize)?
                            .reshape((*t as usize, ()))?,
                    );
                }
                Some(Tensor::cat(&v_thw_vec, 0)?)
            }
            None => None,
        };

        let spatial_merge_size = self.config.vision_config.spatial_merge_size;
        let image_token_id = self.config.image_token_id;
        let video_token_id = self.config.video_token_id;
        let vision_start_token_id = self.config.vision_start_token_id;
        let mut mrope_position_deltas = vec![];
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mask_ = mask
                .cloned()
                .unwrap_or(Tensor::ones_like(&total_input_ids)?)
                .to_device(input_ids.device())?;
            let mut position_ids = Tensor::ones(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut image_index = 0;
            let mut video_index = 0;

            for i in 0..total_input_ids.dim(0)? {
                let mut input_ids_i = total_input_ids.i(i)?;
                let mask_i = mask_.i(i)?;
                // 推理时, attention_mask如果是全1向量,取非0索引的操作没必要
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let nonzero_idx = nonzero_index(&mask_i)?;
                    input_ids_i = input_ids_i.gather(&nonzero_idx, 0)?;
                }
                let mut text_start = 0;
                let mut text_end = 0;
                let mut thw = vec![];
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices =
                    get_vision_next_indices(&input_ids_i, vision_start_token_id as u32);

                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == image_token_id as u32 {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                            }
                            if token == video_token_id as u32 {
                                thw = video_grid_thw
                                    .as_ref()
                                    .unwrap()
                                    .i(video_index)?
                                    .to_vec1::<u32>()?;
                                text_end = vision_indices_vec[j];
                                video_index += 1;
                            }
                            let llm_grid_t = thw[0];
                            let llm_grid_h = thw[1] / spatial_merge_size as u32;
                            let llm_grid_w = thw[2] / spatial_merge_size as u32;
                            let text_len = text_end - text_start;
                            let start_idx = if !llm_pos_ids_list.is_empty() {
                                llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                                    .max_all()?
                                    .to_scalar::<u32>()?
                                    + 1
                            } else {
                                0
                            };
                            let pos_ids = Tensor::arange(
                                start_idx,
                                start_idx + text_len,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                            llm_pos_ids_list.push(pos_ids);

                            let t_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_t,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize * llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let h_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_h,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let w_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_w,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(0)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;

                            let thw_index = Tensor::stack(&[t_index, h_index, w_index], 0)?;
                            llm_pos_ids_list.push(thw_index);
                            text_start = text_end + llm_grid_t * llm_grid_h * llm_grid_w;
                        }
                    }
                    Err(e) => {
                        println!("get vision_indices err: {e}");
                    }
                };
                if text_start < input_ids_i.dim(0)? as u32 {
                    let start_idx = if !llm_pos_ids_list.is_empty() {
                        llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                            .max_all()?
                            .to_scalar::<u32>()?
                            + 1
                    } else {
                        0
                    };
                    let text_len = input_ids_i.dim(0)? as u32 - text_start;
                    let pos_ids =
                        Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                    llm_pos_ids_list.push(pos_ids);
                }
                let llm_position = Tensor::cat(&llm_pos_ids_list, 1)?.reshape((3, 1, ()))?;
                position_ids = position_ids
                    .slice_assign(&[(0..3), (i..i + 1), (0..input_ids.dim(1)?)], &llm_position)?;
                let position_deltas = llm_position.max_all()?.to_scalar::<u32>()? as i64 + 1
                    - input_ids_i.dim(0)? as i64;
                mrope_position_deltas.push(position_deltas);
            }
            let mut mrope_position_deltas = Tensor::new(mrope_position_deltas, input_ids.device())?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids.contiguous()?, mrope_position_deltas))
        } else if let Some(mask) = mask {
            let mut position_ids = mask
                .to_dtype(candle_core::DType::F64)?
                .cumsum(D::Minus1)?
                .to_dtype(candle_core::DType::U32)?
                .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
            for i in 0..position_ids.dim(0)? {
                let mut position_ids_i = position_ids.i(i)?;
                let mask_i = mask.i(i)?;
                // 如果有pad, 将填充位置置为1
                // 当bs>1, 可能存在不同序列长度，需要添加pad使seq_len长度一致
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let zero_indices = zero_index(&mask_i)?;
                    let replace_1 = Tensor::ones(
                        zero_indices.dim(0)?,
                        candle_core::DType::U32,
                        input_ids.device(),
                    )?;
                    position_ids_i = position_ids_i
                        .scatter(&zero_indices, &replace_1, 0)?
                        .unsqueeze(0)?;
                    position_ids = position_ids
                        .slice_assign(&[(i..i + 1), (0..position_ids.dim(1)?)], &position_ids_i)?;
                }
            }
            position_ids = position_ids
                .unsqueeze(0)?
                .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
                .contiguous()?;
            let mut mrope_position_deltas = position_ids
                .max(0)?
                .max(D::Minus1)?
                .broadcast_sub(&Tensor::new(
                    vec![mask.dim(D::Minus1)? as u32 - 1],
                    input_ids.device(),
                )?)?
                .contiguous()?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids, mrope_position_deltas))
        } else {
            let position_ids =
                Tensor::arange(0_u32, input_ids.dim(D::Minus1)? as u32, input_ids.device())?
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(D::Minus1)?))?
                    .contiguous()?;
            let mrope_position_deltas = Tensor::zeros(
                (input_ids.dim(0)?, 1),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            Ok((position_ids, mrope_position_deltas))
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        let mut image_mask = None;
        let mut video_mask = None;
        let mut deepstack_image_embeds = None;
        let mut deepstack_video_embeds = None;
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                let vision_mask = self.get_placeholder_mask(input_ids, true)?;
                let n_image_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;

                // 🌟 [VISION-CACHE] Qwen3VL 은 deepstack 을 실제로 사용하므로
                //    캐시본도 deepstack 까지 완전히 복원되어야 합니다.
                //    복원 개수가 deepstack_visual_indexes 와 다르면 캐시를 폐기합니다.
                let cache_key = crate::models::vision_cache::VisionEmbedCache::compute_key(
                    pixel_values, image_grid_thw
                ).ok();
                let expected_ds = self.visual.deepstack_visual_indexes.len();

                let mut resolved: Option<(Tensor, Vec<Tensor>)> = None;

                if let Some(key) = cache_key {
                    if let Some((cached, cached_ds)) = crate::models::vision_cache::VISION_CACHE
                        .try_load(key, inputs_embeds.device(), inputs_embeds.dtype())
                    {
                        let shape_ok = crate::models::vision_cache::validate_embed_shape(
                            &cached, n_image_tokens as usize
                        ).is_ok();
                        if shape_ok && cached_ds.len() == expected_ds {
                            resolved = Some((cached, cached_ds));
                        } else {
                            println!(
                                "[VISION-CACHE] Discarding stale entry (tokens ok: {}, deepstack {} vs expected {}). Falling back to ViT.",
                                shape_ok, cached_ds.len(), expected_ds
                            );
                        }
                    }
                }

                let (image_embeds, deepstack_img_embed) = match resolved {
                    Some(v) => v,
                    None => {
                        let (split_embeds, ds) =
                            self.get_vision_features(pixel_values, image_grid_thw)?;
                        let merged = Tensor::cat(&split_embeds, 0)?;
                        if let Some(key) = cache_key {
                            if let Err(e) = crate::models::vision_cache::VISION_CACHE
                                .save(key, &merged, &ds)
                            {
                                println!("[VISION-CACHE] Save failed (non-fatal): {}", e);
                            }
                        }
                        (merged, ds)
                    }
                };

                if n_image_tokens as usize != image_embeds.dim(0)? {
                    return Err(anyhow!(format!(
                        "n_image_token num: {} not equal to image_embed len: {}",
                        n_image_tokens,
                        image_embeds.dim(0)?
                    )));
                }
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
                image_mask = Some(vision_mask);
                deepstack_image_embeds = Some(deepstack_img_embed);
            }
        }
        if let Some(pixel_values_video) = pixel_values_video {
            if let Some(video_grid_thw) = video_grid_thw {
                let (video_embeds, deepstack_video_embed) =
                    self.get_vision_features(pixel_values_video, video_grid_thw)?;
                let video_embeds = Tensor::cat(&video_embeds, 0)?;
                let vision_mask = self.get_placeholder_mask(input_ids, false)?;
                let n_video_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
                if n_video_tokens as usize != video_embeds.dim(0)? {
                    return Err(anyhow!(format!(
                        "n_image_token num: {} not equal to image_embed len: {}",
                        n_video_tokens,
                        video_embeds.dim(0)?
                    )));
                }
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
                video_mask = Some(vision_mask);
                deepstack_video_embeds = Some(deepstack_video_embed);
            }
        }
        let mut visual_pos_mask = None;
        let mut deepstack_visual_embeds = None;
        if let Some(image_mask_) = image_mask {
            if let Some(video_mask_) = video_mask {
                let image_mask_ = image_mask_.squeeze(0)?;
                let video_mask_ = video_mask_.squeeze(0)?;
                let visual_mask = bitor_tensor(&image_mask_, &video_mask_)?;
                let visual_none_zero_index = nonzero_index(&visual_mask)?;
                let image_mask_joint: Tensor = image_mask_.gather(&visual_none_zero_index, 0)?;
                let image_nonzero_joint = nonzero_index(&image_mask_joint)?;
                let video_mask_joint: Tensor = video_mask_.gather(&visual_none_zero_index, 0)?;
                let video_nonzero_joint = nonzero_index(&video_mask_joint)?;
                let mut deepstack_embeds = vec![];
                let visual_len = visual_none_zero_index.dim(0)?;
                for (img_embed, vid_embed) in deepstack_image_embeds
                    .unwrap()
                    .iter()
                    .zip(deepstack_video_embeds.unwrap().iter())
                {
                    let embed_joint = Tensor::zeros(
                        (visual_len, img_embed.dim(D::Minus1)?),
                        img_embed.dtype(),
                        img_embed.device(),
                    )?;
                    let embed_joint = embed_joint.index_add(&image_nonzero_joint, img_embed, 0)?;
                    let embed_joint = embed_joint.index_add(&video_nonzero_joint, vid_embed, 0)?;
                    deepstack_embeds.push(embed_joint);
                }
                visual_pos_mask = Some(visual_mask.unsqueeze(0)?);
                deepstack_visual_embeds = Some(deepstack_embeds);
            } else {
                visual_pos_mask = Some(image_mask_);
                deepstack_visual_embeds = deepstack_image_embeds;
            }
        } else if let Some(video_mask_) = video_mask {
            visual_pos_mask = Some(video_mask_);
            deepstack_visual_embeds = deepstack_video_embeds;
        }

        let position_ids;
        let rope_deltas;
        if (cache_position.is_some() && cache_position.unwrap().i(0)?.to_scalar::<u32>()? == 0)
            || self.rope_deltas.is_none()
        {
            (position_ids, rope_deltas) =
                self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if let Some(cache_position) = cache_position {
                if let Some(rope_deltas) = &self.rope_deltas {
                    cache_position
                        .i(0)?
                        .to_dtype(rope_deltas.dtype())?
                        .broadcast_add(rope_deltas)?
                        .contiguous()?
                        .to_dtype(candle_core::DType::U32)?
                } else {
                    Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
                }
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?;
        }
        
        let total_len = inputs_embeds.dim(1)?;
        let chunk_size = 256;
        let mut final_hidden_state = None;

        if total_len > 1 {
            let mut processed = 0;
            let mut current_offset = seqlen_offset;
            let mut vision_offset = 0;

            while processed < total_len {
                let take = (total_len - processed).min(chunk_size);
                let chunk_embeds = inputs_embeds.narrow(1, processed, take)?;
                let chunk_pos_ids = position_ids.narrow(2, processed, take)?;

                let mut chunk_visual_pos_mask = None;
                let mut chunk_deepstack_embeds = None;

                // 🌟 [VISION ALIGNMENT] 딥스택 비전 텐서도 현재 청크에 맞춰서 슬라이싱합니다.
                if let Some(v_mask) = visual_pos_mask.as_ref() {
                    let c_mask = v_mask.narrow(1, processed, take)?;
                    let num_vision_in_chunk = c_mask.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;

                    if num_vision_in_chunk > 0 {
                        chunk_visual_pos_mask = Some(c_mask.clone());
                        let mut sliced_deepstacks = Vec::new();
                        for ds in deepstack_visual_embeds.as_ref().unwrap() {
                            let sliced_ds = ds.narrow(0, vision_offset, num_vision_in_chunk)?;
                            sliced_deepstacks.push(sliced_ds);
                        }
                        chunk_deepstack_embeds = Some(sliced_deepstacks);
                    }
                    vision_offset += num_vision_in_chunk;
                }

                let outputs = self.language_model.forward(
                    &chunk_embeds,
                    current_offset,
                    Some(&chunk_pos_ids),
                    chunk_visual_pos_mask.as_ref(),
                    chunk_deepstack_embeds,
                )?;

                // (레이어 내부에서 자동 압축되므로 밖에서 호출할 필요 없음)

                if processed + take == total_len {
                    let seq_len = outputs.dim(1)?;
                    final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
                }

                // 🌟 [VRAM/RAM 최적화]
                if inputs_embeds.device().is_cuda() {
                    let _ = inputs_embeds.device().synchronize();
                }

                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }

                processed += take;
                current_offset += take;

                use std::io::Write;
                print!("\r[Qwen3VL-PREFILL] {} / {} tokens processed", processed, total_len);
                let _ = std::io::stdout().flush();
            }
            println!("\n[Qwen3VL-PREFILL] Complete. Starting Generation...");
        } else {
            let outputs = self.language_model.forward(
                &inputs_embeds,
                seqlen_offset,
                Some(&position_ids),
                visual_pos_mask.as_ref(),
                deepstack_visual_embeds,
            )?;
            
            // (레이어 내부에서 자동 압축되므로 밖에서 호출할 필요 없음)
            
            let seq_len = outputs.dim(1)?;
            final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
        }

        let hidden_state = final_hidden_state.unwrap();
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    pub fn evacuate_kv_to_cpu(&mut self) -> Result<()> {
        self.language_model.evacuate_kv_to_cpu()
    }

    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.language_model.get_kv_cache()
    }

    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        self.language_model.set_kv_cache(cache)
    }

    // 🌟 [추가] Semantic Bias 연산을 위해 전체 단어장의 벡터(Weight)를 그대로 반환합니다.
    pub fn get_embed_tokens(&self) -> Tensor {
        self.language_model.get_embed_tokens()
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.embedding_token_id(input_ids)
    }
}
