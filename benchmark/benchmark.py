import torch
import qai_hub as hub

def run_compile(traced_model, device, input_specs):
    compile_job = hub.submit_compile_job(
       model = traced_model,
       device = device,
       input_specs = input_specs
    )

    assert isinstance(compile_job, hub.CompileJob)
    return compile_job

def run_profile(compiled_job, device):
    profile_job = hub.submit_profile_job(
      model = compiled_job.get_target_model(),
      device = device,
      name = compiled_job.name + "_profiling"
    )

    assert isinstance(profile_job, hub.ProfileJob)
    return profile_job

def get_traced_model(model, example_inputs, strict=False):
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs, strict=strict)
    return traced_model




