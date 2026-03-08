import torch
import torch.nn as nn
from liquid_audio.model.lfm2_audio import LFM2AudioModel
import argparse
import os
from benchmark.benchmark import get_traced_model, run_compile, run_profile
from benchmark.extract_metrices import extract_and_print_metrics
import qai_hub as hub
import wandb

parser = argparse.ArgumentParser('Benchmark LFM2-Audio on Qualcomm AI Hub with demo input',)

parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
parser.add_argument('--duration', type=float, default=10, help='Audio duration in seconds for demo input (default: 3.0)')
parser.add_argument('--channels', type=int, default=1, help='Number of audio channels')
parser.add_argument('--device', default='Samsung Galaxy S24', help='Target Qualcomm device name')
parser.add_argument('--wandb_project', default='LFM2-Audio-Benchmark', help='WandB project name')
parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB logging mode')

args, _ = parser.parse_known_args()


class UnifiedSpeechToSpeech(nn.Module):
    """
    This is needed because Qualcomm AI Hub requires models that can be
    traced with TorchScript (no dynamic loops or conditionals).
    """
    
    def __init__(self, conformer, audio_adapter, lfm):
        super().__init__()
        self.conformer = conformer
        self.audio_adapter = audio_adapter
        self.lfm = lfm
    
    def forward(self, mel_spectrogram, mel_lengths):
        # Step 1: Audio encoding
        conformer_out = self.conformer(mel_spectrogram, mel_lengths)
        
        # Step 2: Align to LFM embedding space
        adapted = self.audio_adapter(conformer_out)
        
        # Step 3: Process through language model
        lfm_out = self.lfm(adapted)
        
        return lfm_out


def create_dummy_input(duration_sec=args.duration, batch_size=1):
    n_mels = 128  # LFM2-Audio uses 128 mel bins
    frames = int(duration_sec * 50)  # ~50 fps for mel spectrograms
    
    mel_spectrogram = torch.randn(batch_size, n_mels, frames)
    mel_lengths = torch.tensor([frames] * batch_size)
    
    return mel_spectrogram, mel_lengths


def load_and_prepare_model():
        
    print("Loading LFM2AudioModel from HuggingFace...")
    full_model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        device="cpu"  
    )
    full_model.eval()
    
    unified_model = UnifiedSpeechToSpeech(
        full_model.conformer,
        full_model.audio_adapter,
        full_model.lfm
    )
    unified_model.eval()
    
    return unified_model


def trace_model(model, dummy_inputs):
    mel_spectrogram, mel_lengths = dummy_inputs
    
    print("Tracing model with TorchScript...")
    traced_model = get_traced_model(
        model=model,
        example_inputs=(mel_spectrogram, mel_lengths),
        strict=False
    )
    
    return traced_model


def main():    
    
    # Initialize WandB
    if args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            name=f"{args.device}_duration:{args.duration}s",
            mode=args.wandb_mode,
            config=vars(args)
        )
    
    # Load model
    unified_model = load_and_prepare_model()
    
    # Create dummy input
    dummy_inputs = create_dummy_input(args.duration)
    mel_spectrogram, mel_lengths = dummy_inputs
    
    #Trace model
    traced_model = trace_model(unified_model, dummy_inputs)
    
    # Use benchmark.run_compile with custom input specs
    input_specs = {
        "mel_spectrogram": tuple(mel_spectrogram.shape),
        "mel_lengths": tuple(mel_lengths.shape)
    }
    
    compile_job = run_compile(
        traced_model=traced_model,
        device=hub.Device(args.device),
        input_specs=input_specs
    )
    
    profile_job = run_profile(
        compiled_job=compile_job,
        device=hub.Device(args.device)
    )
    
    profile_data = profile_job.download_profile()
    
    # Download results to disk
    results_dir = os.path.join(os.getcwd(), 'LFM2-Audio-Results')
    profile_job.download_results(results_dir)
    print(f"Results saved to: {results_dir}")
    
    # Extract metrics using benchmark function (returns dict, no printing)
    metrics = extract_and_print_metrics(profile_data)
    
    # Print summary
    inference_time_ms = metrics.get('estimated_inference_time_ms', 0)
    rtf = (args.duration * 1000) / inference_time_ms if inference_time_ms > 0 else 0
    
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Inference Time:        {inference_time_ms:.2f} ms")
    print(f"Peak Memory:           {metrics.get('estimated_inference_peak_memory_mb', 0):.2f} MB")
    print(f"Real-time Factor:      {rtf:.2f}x")
    print(f"Throughput:            {metrics.get('throughput_fps', 0):.2f} FPS")
    print()
    print("Mobile Criteria Check:")
    print(f"  Real-time:           {'✓ YES' if rtf >= 1.0 else '✗ NO'} (≥1.0x)")
    print(f"  Low latency:         {'✓ YES' if inference_time_ms <= 500 else '✗ NO'} (≤500ms)")
    print(f"  Memory efficient:    {'✓ YES' if metrics.get('estimated_inference_peak_memory_mb', 0) <= 4000 else '⚠ HIGH'} (≤4GB)")
    print("=" * 80)
    
    print("\n✓ Benchmark complete!")
    
    # Prepare complete metrics for WandB
    complete_metrics = {
        'model_name': 'LFM2-Audio',
        'model_version': 'LFM2.5-Audio-1.5B',
        'device': args.device,
        'audio_duration_sec': args.duration,
        'n_mels': 128,
        'frames': int(args.duration * 50),
        'input_mel_shape': str(mel_spectrogram.shape),
        'input_mel_lengths_shape': str(mel_lengths.shape),
        **metrics
    }
    
    if args.wandb_mode != 'disabled':    
        wandb.log(complete_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
