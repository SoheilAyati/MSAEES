#!/usr/bin/env python3
"""
app.py  --  Streamlit control panel for the MS2 pipeline
========================================================

One UI for all three scripts: generate a corpus, train a model, and run
inference -- with the result graphs shown inline so they're easy to view and
present. Each inference run is saved to its own timestamped folder.

Run it (from this folder):

    pip install streamlit          # or: uv pip install streamlit
    streamlit run app.py

It is a thin wrapper: every button just runs the same generate_corpus.py /
train.py / infer.py you use on the command line, then displays the outputs.
"""
from __future__ import annotations
import os, sys, glob, json, subprocess, datetime

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
AGG_SCRIPT = os.path.join(REPO, "Scripts", "Aggregator", "mix_measured_scenarios.py")


# --------------------------------------------------------------------------
def run_cmd(cmd, spinner="Running..."):
    """Run a subprocess in the pipeline folder; show command + output."""
    st.caption("command")
    st.code(" ".join(str(c) for c in cmd), language="bash")
    with st.spinner(spinner):
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.stdout:
        st.text(r.stdout[-4000:])
    if r.returncode != 0:
        st.error("Failed (exit %d)" % r.returncode)
        st.text((r.stderr or "")[-3000:])
        return False
    return True


def find(patterns):
    out = []
    for p in patterns:
        out += glob.glob(p, recursive=True)
    return sorted(set(out))


def show_images(folder):
    imgs = sorted(glob.glob(os.path.join(folder, "*.png")))
    for img in imgs:
        st.image(img, caption=os.path.basename(img), use_container_width=True)
    return imgs


# --------------------------------------------------------------------------
st.set_page_config(page_title="NILM MS2 Pipeline", layout="wide")
st.title("NILM Pipeline")

tab_infer, tab_train, tab_gen, tab_agg = st.tabs(
    ["Infer", "Train", "Generate corpus", "Aggregate (measured)"])

# ---------------- INFER ----------------
with tab_infer:
    st.subheader("Run a trained model on one signal file")
    signal_choices = find([
        os.path.join(REPO, "Pre_Measured", "*.csv"),
        os.path.join(REPO, "Synthetic_Data", "Mixed", "*.h5"),
        os.path.join(REPO, "Synthetic_Data", "Single", "*.h5"),
        os.path.join(HERE, "corpus", "scenario_*.h5"),
    ])
    model_choices = find([os.path.join(HERE, "output", "**", "*.joblib")])

    c1, c2 = st.columns(2)
    with c1:
        sig_pick = st.selectbox("Signal file", ["(type a path below)"] + signal_choices)
        sig_path = st.text_input("…or path", value="" if sig_pick == "(type a path below)" else sig_pick)
        sig_path = sig_path or (sig_pick if sig_pick != "(type a path below)" else "")
    with c2:
        mdl_pick = st.selectbox("Model (.joblib)", ["(type a path below)"] + model_choices)
        mdl_path = st.text_input("…or model path", value="" if mdl_pick == "(type a path below)" else mdl_pick)
        mdl_path = mdl_path or (mdl_pick if mdl_pick != "(type a path below)" else "")

    cc1, cc2 = st.columns(2)
    win = cc1.number_input("Window (s)", value=10.0, min_value=1.0, key="iwin")
    stride = cc2.number_input("Stride (s)", value=5.0, min_value=1.0, key="istride")

    if st.button("Run inference", type="primary"):
        if not sig_path or not mdl_path:
            st.warning("Pick a signal file and a model.")
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.splitext(os.path.basename(sig_path))[0]
            outdir = os.path.join("output", f"infer_{stem}_{ts}")        # unique each run
            cmd = [PY, "infer.py", "--input", sig_path, "--model", mdl_path,
                   "--window", str(win), "--stride", str(stride), "--out", outdir]
            if run_cmd(cmd, "Running inference..."):
                full = os.path.join(HERE, outdir)
                st.success("Saved to %s" % outdir)
                show_images(full)
                sj = os.path.join(full, "summary.json")
                if os.path.exists(sj):
                    st.subheader("Summary")
                    st.json(json.load(open(sj)))

# ---------------- TRAIN ----------------
with tab_train:
    st.subheader("Train a model")
    task = st.radio("Task", ["identify", "disaggregate", "presence"], horizontal=True,
                    help="identify = window→appliance (single-device); "
                         "disaggregate = aggregate→per-appliance power; "
                         "presence = which appliances are ON per window (multi-label)")
    default_data = ("corpus/scenario_*.h5" if task in ("disaggregate", "presence")
                    else os.path.join(REPO, "Synthetic_Data", "Single"))
    data = st.text_input("Training data (folder, glob, or files)", value=default_data)
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", ["rf", "lgbm", "mlp"],
                         help="mlp = neural net on raw waveform (disaggregate/presence)")
    feats = c2.selectbox("Features (identify)", ["auto", "common", "full"])
    outdir = c3.text_input("Model out dir", value="output")
    c4, c5 = st.columns(2)
    twin = c4.number_input("Window (s)", value=30.0, min_value=1.0, key="twin")
    tstride = c5.number_input("Stride (s)", value=30.0, min_value=1.0, key="tstride")

    if st.button("Train model", type="primary"):
        cmd = [PY, "train.py", "--task", task, "--data", *data.split(),
               "--model", model, "--features", feats,
               "--window", str(twin), "--stride", str(tstride), "--out", outdir]
        if run_cmd(cmd, "Training..."):
            st.success("Model saved in %s" % outdir)
            suffix = "_mlp" if model == "mlp" else ""
            mj = os.path.join(HERE, outdir, f"train_{task}{suffix}_metrics.json")
            if os.path.exists(mj):
                st.subheader("Held-out metrics")
                st.json(json.load(open(mj)))
            cm = os.path.join(HERE, outdir, "train_identify_confusion.png")
            if task == "identify" and os.path.exists(cm):
                st.image(cm, caption="Confusion matrix (held-out)", use_container_width=True)

# ---------------- GENERATE ----------------
with tab_gen:
    st.subheader("Generate a multi-seed synthetic corpus")
    seeds = st.text_input("Seeds (space-separated)", value="101 102 103")
    gout = st.text_input("Output dir", value="corpus")
    dur = st.number_input("Duration per day (s)", value=86400, min_value=3600, step=3600)
    st.caption("~230 MB per seed (24 h). Use fewer seeds or a shorter duration for a quick test.")
    if st.button("Generate corpus", type="primary"):
        cmd = [PY, "generate_corpus.py", "--seeds", *seeds.split(),
               "--outdir", gout, "--duration", str(int(dur))]
        if run_cmd(cmd, "Generating (this can take a minute per seed)..."):
            scen = find([os.path.join(HERE, gout, "scenario_*.h5")])
            st.success("Corpus ready: %d scenario(s) in %s" % (len(scen), gout))
            st.write([os.path.basename(s) for s in scen])

# ---------------- AGGREGATE (measured) ----------------
with tab_agg:
    st.subheader("Mix real PAC4200 recordings into ground-truth scenarios")
    st.caption("Converts each single-appliance recording to per-appliance format, "
               "loops it to a common length, then runs the aggregator to build "
               "scenario .h5 files with /ground_truth. Use them under Train -> "
               "disaggregate / presence.")
    rec_dir = st.text_input(
        "Recordings folder",
        value=os.path.join(REPO, "Scripts", "PAC4200_reader", "recordings"))
    agg_out = st.text_input(
        "Output dir",
        value=os.path.join(REPO, "Scripts", "Aggregator", "measured_scenarios"))
    a1, a2, a3 = st.columns(3)
    n_scen = a1.number_input("Scenarios", value=6, min_value=1, step=1, key="an")
    adur = a2.number_input("Duration (s)", value=300.0, min_value=10.0, step=10.0, key="adur")
    aseed = a3.number_input("Seed", value=0, min_value=0, step=1, key="aseed")
    a4, a5, a6 = st.columns(3)
    min_app = a4.number_input("Min appliances", value=2, min_value=2, step=1, key="amin")
    max_app = a5.number_input("Max appliances", value=4, min_value=2, step=1, key="amax")
    do_plot = a6.checkbox("Decomposition plots", value=True, key="aplot")

    if st.button("Aggregate measured recordings", type="primary"):
        if not os.path.isfile(AGG_SCRIPT):
            st.error("Aggregator script not found: %s" % AGG_SCRIPT)
        elif not os.path.isdir(rec_dir):
            st.warning("Recordings folder not found: %s" % rec_dir)
        else:
            cmd = [PY, AGG_SCRIPT, "--recordings", rec_dir, "--out", agg_out,
                   "--n-scenarios", str(int(n_scen)), "--duration", str(adur),
                   "--min-app", str(int(min_app)), "--max-app", str(int(max_app)),
                   "--seed", str(int(aseed))]
            if do_plot:
                cmd.append("--plot")
            if run_cmd(cmd, "Aggregating measured recordings..."):
                scen = find([os.path.join(agg_out, "measured_scenario_*.h5")])
                st.success("Built %d scenario(s) in %s" % (len(scen), agg_out))
                st.write([os.path.basename(s) for s in scen])
                mf = os.path.join(agg_out, "manifest.json")
                if os.path.exists(mf):
                    st.subheader("Manifest")
                    st.json(json.load(open(mf)))
                show_images(agg_out)

st.divider()
st.caption("Outputs are written under MS2_Pipeline/output/. Each inference run "
           "gets its own timestamped folder.")
