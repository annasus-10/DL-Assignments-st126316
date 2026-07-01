"""A6: Speech Processing — unified CLI entrypoint.

Examples:
    python3 run.py --model ctc --epochs 300 --train
    python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train
    python3 run.py --model voice-clone --extract-se --reference my_voice.wav
    python3 run.py --model voice-clone --accent us --text "I got the job!" --generate
    python3 run.py --model voice-clone --accent all --text "Hello world" --generate
"""
import argparse
import os

import torch


def run_ctc(args):
    from models.ctc_model import train_ctc_model, greedy_decode_grid

    frames_per_char = tuple(args.frames_per_char) if args.frames_per_char else (3, 8)
    model, losses, cers = train_ctc_model(n_steps=args.epochs, frames_per_char=frames_per_char, lr=args.lr)

    print(f'Final loss: {losses[-1]:.4f}')
    print(f'Final CER: {cers[-1]:.4f}')
    below_10 = next((i for i, c in enumerate(cers) if c < 0.10), None)
    print(f'First step CER < 10%: {below_10}')

    if args.save_figures:
        os.makedirs('outputs/figures', exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(losses)
        axes[0].set_title('CTC Loss vs Training Step')
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('Loss')
        axes[1].plot(cers)
        axes[1].axhline(0.10, color='red', linestyle='--', label='10% threshold')
        axes[1].set_title('Character Error Rate vs Training Step')
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('CER')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('outputs/figures/ctc_cer_curve.png', dpi=150, bbox_inches='tight')

        greedy_decode_grid(model, frames_per_char=frames_per_char,
                            save_path='outputs/figures/ctc_greedy_decode.png')
        print('Saved figures to outputs/figures/')

    os.makedirs('saved', exist_ok=True)
    torch.save(model.state_dict(), 'saved/ctc_model.pt')
    print('Saved model to saved/ctc_model.pt')


def run_wav2vec2_probe(args):
    from utils.wav2vec_probe import (load_speechcommands_subset, extract_wav2vec2_features,
                                      extract_mel_features, train_linear_probe, plot_tsne)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe_words = args.classes.split(',')

    by_label = load_speechcommands_subset(probe_words, n_per_class=args.n_per_class)
    X_w2v, y_w2v = extract_wav2vec2_features(by_label, probe_words, device=device)
    X_mel, y_mel = extract_mel_features(by_label, probe_words)

    acc_w2v, _ = train_linear_probe(X_w2v, y_w2v, n_classes=len(probe_words))
    acc_mel, _ = train_linear_probe(X_mel, y_mel, n_classes=len(probe_words))

    print(f'wav2vec2 linear probe test acc: {acc_w2v*100:.1f}%')
    print(f'raw mel linear probe test acc:  {acc_mel*100:.1f}%')
    print(f'random baseline: {100/len(probe_words):.1f}%')

    if args.save_figures:
        os.makedirs('outputs/figures', exist_ok=True)
        plot_tsne(X_w2v, y_w2v, probe_words,
                  'Frozen wav2vec2 Embeddings (t-SNE)\nNo transcripts used during pretraining',
                  save_path='outputs/figures/wav2vec2_tsne.png')
        plot_tsne(X_mel, y_mel, probe_words,
                  'Raw Mel-Spectrogram Features (t-SNE)\nNo pretraining, mean-pooled',
                  save_path='outputs/figures/mel_baseline_tsne.png')
        print('Saved figures to outputs/figures/')


def run_voice_clone(args):
    from models.voice_clone import load_tone_color_converter, extract_target_se, synthesize_accent, get_device

    device = get_device()

    if args.extract_se:
        tone_color_converter, _ = load_tone_color_converter(device)
        ref_path = f'data/voice_clone/{args.reference}'
        os.makedirs('saved', exist_ok=True)
        target_se = extract_target_se(ref_path, tone_color_converter, save_path='saved/target_se.pt')
        print(f'Extracted tone color embedding: shape {target_se.shape}')
        print('Saved to saved/target_se.pt')
        return

    if args.generate:
        tone_color_converter, ckpt_dir = load_tone_color_converter(device)
        target_se = torch.load('saved/target_se.pt', map_location=device)

        accents = ['us', 'br', 'india', 'au'] if args.accent == 'all' else [args.accent]
        for accent in accents:
            base_path, out_path = synthesize_accent(
                args.text, accent, target_se, tone_color_converter, ckpt_dir, device=device)
            print(f'[{accent:6}] base={base_path} -> cloned={out_path}')
        return

    print('Specify --extract-se or --generate for voice-clone mode.')


def main():
    parser = argparse.ArgumentParser(description='A6: Speech Processing CLI')
    parser.add_argument('--model', required=True, choices=['ctc', 'wav2vec2-probe', 'voice-clone'])

    # CTC args
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--frames-per-char', type=int, nargs=2, default=None,
                         help='e.g. --frames-per-char 1 2')
    parser.add_argument('--train', action='store_true')

    # wav2vec2-probe args
    parser.add_argument('--dataset', default='speechcommands')
    parser.add_argument('--classes', default='yes,no,stop,go')
    parser.add_argument('--n-per-class', type=int, default=40)

    # voice-clone args
    parser.add_argument('--extract-se', action='store_true')
    parser.add_argument('--reference', default='my_voice.wav')
    parser.add_argument('--accent', default='us', choices=['us', 'br', 'india', 'au', 'all'])
    parser.add_argument('--text', default='I got the job!')
    parser.add_argument('--generate', action='store_true')

    # shared
    parser.add_argument('--save-figures', action='store_true', default=True)

    args = parser.parse_args()

    if args.model == 'ctc':
        run_ctc(args)
    elif args.model == 'wav2vec2-probe':
        run_wav2vec2_probe(args)
    elif args.model == 'voice-clone':
        run_voice_clone(args)


if __name__ == '__main__':
    main()
