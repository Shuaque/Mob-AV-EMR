import os
import cv2
import math
import numpy as np
import torch
import torchaudio
from pathlib import Path
from fairseq import checkpoint_utils, tasks, utils

# ====== 你需要先准备这三样路径 ======
CKPT = "/workspace/shuaque/Mob-AV-EMR/pretrained/avhubert/base_vox_iter5.pt"     # 预训练或下游ASR微调的 ckpt
WAV  = "/workspace/shuaque/Data/corpus/LRW-1000-CORPUS/audio/audio/0a00ac918bf88c27735c8caf2b48f529.wav"        # 从视频里抽出的 16kHz 单声道 wav
FRAMES_DIR = "/workspace/shuaque/Data/corpus/LRW-1000-CORPUS/image/images/0a00ac918bf88c27735c8caf2b48f529"     # 口型 ROI 帧目录, 命名 000001.png, 000002.png, ...

# ====== 基础超参（与训练保持一致） ======
IMG_SIZE = 88
IMG_MEAN = 0.421
IMG_STD  = 0.165
FPS = 25

def load_audio_16k(wav_path):
    wav, sr = torchaudio.load(wav_path)  # [ch, T]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # 转单声道
    return wav.squeeze(0)  # [T]

def load_roi_frames(frames_dir, to_gray=True):
    frame_files = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # H,W
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # H,W,3
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if to_gray:
            img = (img.astype(np.float32)/255.0 - IMG_MEAN) / IMG_STD  # 归一化
            img = img[None, ...]  # C=1
        else:
            img = img.astype(np.float32)/255.0
            img = (img - IMG_MEAN) / IMG_STD
            img = np.transpose(img, (2,0,1))  # C,H,W
        frames.append(img)
    # 形状: [T, C, H, W] -> torch [1, T, C, H, W]
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).unsqueeze(0)

@torch.no_grad()
def main():
    utils.import_user_module(None)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([CKPT])
    model = models[0].eval().cuda()

    # 准备输入
    wav = load_audio_16k(WAV).cuda()                # [T_audio]
    video = load_roi_frames(FRAMES_DIR).cuda()      # [1, T_v, C, H, W]
    B, T_v, C, H, W = video.shape

    # padding mask（True 表示需要 mask）
    audio_padding_mask = torch.zeros(1, wav.shape[-1], dtype=torch.bool, device=wav.device)
    video_padding_mask = torch.zeros(1, T_v, dtype=torch.bool, device=wav.device)

    # 有些 AV-HuBERT 实现支持 features_only/mask 等开关；这里尽量兼容两种常见接口：
    # 方式1：直接调用模型的 extract_features（若存在）
    if hasattr(model, "extract_features"):
        out = model.extract_features(
            source=wav.unsqueeze(0),                   # [B, T_audio]
            padding_mask=audio_padding_mask,          # [B, T_audio]
            video=video,                              # [B, T_v, C, H, W]
            video_padding_mask=video_padding_mask,    # [B, T_v]
            mask=False,
            features_only=True,
            return_all_hiddens=True,
        )
        # 兼容不同返回风格
        if isinstance(out, (list, tuple)):
            feats = out[0]            # 取最终层特征 [B, T', C]
            all_hs = out[-1] if len(out) > 1 else None
        elif isinstance(out, dict) and "x" in out:
            feats = out["x"]          # [B, T', C]
            all_hs = out.get("hs", None)
        else:
            feats = out
            all_hs = None
    else:
        # 方式2：走 forward 并从 encoder_out 中拿（Fairseq 常见字典结构）
        # 注意：不同分支/版本字段名略有差异，按需调整 'encoder_out'
        net_input = {
            "source": wav.unsqueeze(0),                  # [B, T_audio]
            "padding_mask": audio_padding_mask,          # [B, T_audio]
            "video": video,                              # [B, T_v, C, H, W]
            "video_padding_mask": video_padding_mask,    # [B, T_v]
        }
        out = model(**net_input)
        # 常见结构：out["encoder_out"] : [T', B, C]
        if isinstance(out, dict) and "encoder_out" in out:
            enc = out["encoder_out"]                    # 通常是 length-1 list
            if isinstance(enc, (list, tuple)):
                enc = enc[0]
            feats = enc.transpose(0,1)                  # -> [B, T', C]
            all_hs = None
        else:
            raise RuntimeError("无法从模型返回中找到 encoder 表征，请检查你的 AV-HuBERT 分支。")

    print("Frame-level feats:", feats.shape)  # [B, T', C]  ~ T'≈T_v（对齐到25Hz）
    # 例如导出逐帧表征（去掉 batch 维）
    np.save("avhubert_feats.npy", feats[0].float().cpu().numpy())

    # 也可做池化得到整段向量
    clip_repr = feats.mean(dim=1)  # [B, C]
    np.save("avhubert_clip_repr.npy", clip_repr[0].float().cpu().numpy())
    print("Done. Saved to avhubert_feats.npy / avhubert_clip_repr.npy")

if __name__ == "__main__":
    main()