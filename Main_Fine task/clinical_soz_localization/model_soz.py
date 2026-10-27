
from pathlib import Path

import inspect

import torch



import downstream.seizure_detection.model_seizure as model_module





def build_soz_model(cfg):

    model_cfg = dict(cfg["model"])

    data_cfg = cfg["data"]



    seq_len = int(float(data_cfg["context_sec"]) / float(data_cfg.get("patch_sec", 1)))

    max_channels = int(data_cfg.get("max_channels", 128))



    base_kwargs = dict(model_cfg)

    base_kwargs["seq_len"] = seq_len

    base_kwargs["max_channels"] = max_channels



    candidate_names = [

        "SABrantSeizureClassifier",

        "SeizureClassifier",

        "SeizureDetectionModel",

        "SeizureModel",

        "SA_Brant_SeizureClassifier",

    ]



    for name in candidate_names:

        if not hasattr(model_module, name):

            continue



        cls = getattr(model_module, name)



        trials = []



        tmp = dict(base_kwargs)

        tmp["num_classes"] = 2

        trials.append(tmp)



        tmp = dict(base_kwargs)

        tmp["n_classes"] = 2

        trials.append(tmp)



        tmp = dict(base_kwargs)

        tmp["n_outputs"] = 2

        trials.append(tmp)



        trials.append(dict(base_kwargs))

        trials.append(dict(model_cfg))



        for kwargs in trials:

            try:

                sig = inspect.signature(cls.__init__)

                params = sig.parameters

                has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())



                if has_var_kw:

                    filtered = kwargs

                else:

                    allowed = set(params.keys()) - {"self"}

                    filtered = {k: v for k, v in kwargs.items() if k in allowed}



                model = cls(**filtered)



                print("Built SOZ model using:", name)

                print("Model args:", filtered)

                return model



            except Exception:

                continue



    print("\nCould not build SOZ model.")

    print("Available classes in downstream/seizure_detection/model_seizure.py:")

    for n, obj in inspect.getmembers(model_module):

        if inspect.isclass(obj):

            print(" ", n)



    raise RuntimeError(

        "Could not build model. Run: grep -n \"class SABrantSeizureClassifier\\|def __init__\" downstream/seizure_detection/model_seizure.py"

    )





def get_state_dict(ckpt):

    if isinstance(ckpt, dict):

        for key in ["model", "model_state_dict", "state_dict", "net", "network"]:

            if key in ckpt:

                return ckpt[key]

    return ckpt







def load_checkpoint_shape_safe(model, ckpt_path, device="cpu"):

    ckpt_path = Path(ckpt_path)

    print("Loading checkpoint:", ckpt_path)



    ckpt = torch.load(str(ckpt_path), map_location=device)

    sd = get_state_dict(ckpt)



    model_sd = model.state_dict()

    loadable = {}

    skipped = []



    # Try direct key and common wrapper prefixes used by classifier models

    prefixes = ["", "encoder.", "backbone.", "mae.", "brant.", "foundation.", "model."]



    for k, v in sd.items():

        kk0 = k

        if kk0.startswith("module."):

            kk0 = kk0[len("module."):]



        matched = False



        for pref in prefixes:

            kk = pref + kk0



            if kk in model_sd and hasattr(v, "shape") and tuple(v.shape) == tuple(model_sd[kk].shape):

                loadable[kk] = v

                matched = True

                break



        if not matched:

            skipped.append(k)



    missing, unexpected = model.load_state_dict(loadable, strict=False)



    print("Loaded tensors:", len(loadable))

    print("Skipped tensors:", len(skipped))

    print("Missing keys:", len(missing))

    print("Unexpected keys:", len(unexpected))



    if len(loadable) > 0:

        print("First loaded keys:", list(loadable.keys())[:10])



    if len(skipped) > 0:

        print("First skipped keys:", skipped[:10])



    if len(loadable) == 0:

        print("\\nWARNING: No pretrained tensors were loaded.")

        print("This means the checkpoint key names still do not match the classifier model.")

        print("Run this diagnostic:")

        print("python - <<'DIAG'")

        print("import torch")

        print("from downstream.clinical_soz_localization.model_soz import build_soz_model")

        print("import yaml")

        print("cfg=yaml.safe_load(open('downstream/clinical_soz_localization/configs/soz_omni_seeg_60s.yaml'))")

        print("m=build_soz_model(cfg)")

        print("print(list(m.state_dict().keys())[:30])")

        print("ck=torch.load('/home/ubuntu/IJAZ/SEEG datast/pretrainingdataset/runs/scale_aligned_brant_edf_stream_real/final_pretrained.pt', map_location='cpu')")

        print("sd=ck.get('model', ck) if isinstance(ck, dict) else ck")

        print("print(list(sd.keys())[:30])")

        print("DIAG")



    return model


def logits_from_output(out):

    if isinstance(out, tuple):

        out = out[0]



    if isinstance(out, dict):

        for k in ["logits", "out", "prediction", "pred"]:

            if k in out:

                return out[k]



    return out

