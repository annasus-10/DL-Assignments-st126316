"""Voice cloning: extract a tone color embedding from a reference clip, then
synthesize text in a given accent via MeloTTS and convert it to the target voice."""
import os

import torch
from huggingface_hub import snapshot_download

STYLE_TO_SE = {
    'us':    ('en-us.pth',    'EN-US'),
    'br':    ('en-br.pth',    'EN-BR'),
    'india': ('en-india.pth', 'EN_INDIA'),
    'au':    ('en-au.pth',    'EN-AU'),
}


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_tone_color_converter(device=None):
    from openvoice.api import ToneColorConverter

    device = device or get_device()
    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2')
    converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=device)
    converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    return converter, ckpt_dir


def extract_target_se(reference_path, tone_color_converter, save_path=None, vad=True):
    """Extract a tone color embedding from a reference voice clip."""
    from openvoice import se_extractor

    target_se, audio_name = se_extractor.get_se(
        reference_path, tone_color_converter,
        target_dir='data/voice_clone/processed', vad=vad)
    if save_path:
        torch.save(target_se, save_path)
    return target_se


def synthesize_accent(text, accent, target_se, tone_color_converter, ckpt_dir,
                       out_dir='outputs/audio', device=None, speed=1.0, tau=0.3):
    """Synthesize `text` in MeloTTS's base `accent` voice, then convert to target_se's tone color."""
    from melo.api import TTS

    device = device or get_device()
    os.makedirs('data/voice_clone', exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    se_file, spk_key = STYLE_TO_SE[accent]

    base_tts = TTS(language='EN', device=device)
    speaker_ids = dict(base_tts.hps.data.spk2id)
    spk_id = speaker_ids.get(spk_key, speaker_ids.get('EN-US'))

    base_path = f'data/voice_clone/base_{accent}.wav'
    out_path = f'{out_dir}/cloned_{accent}.wav'

    base_tts.tts_to_file(text, spk_id, base_path, speed=speed)

    source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=device)
    tone_color_converter.convert(
        audio_src_path=base_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=out_path,
        tau=tau,
    )
    return base_path, out_path
