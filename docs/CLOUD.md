# Training on a cloud GPU (Kaggle, RunPod)

Almost all the training time goes into the network updates: batch 2048 through 3x512 twin critics, one
gradient step per environment step. A CPU core manages about 3 steps/s; a GPU manages roughly 50 or more
(the tutorial logged 48 on its GPU). A 100k-step run therefore drops from about 9 h to roughly 30-40 min.

`run_all.py` does the whole pipeline on any machine:
1. trains the jobs in parallel, spreading them over the GPUs (each run resumes from its checkpoint);
2. evaluates each finished run on 100 test episodes;
3. records the videos;
4. plots the learning curves;
5. writes `results/summary.md`.

```bash
python run_all.py --jobs panda_ddpg:0,1,2 ur5e_ddpg:0 --steps 100000 --parallel 4
```

## Kaggle (free, about 30 GPU hours a week)

1. Download [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb).
2. On kaggle.com, go to **Create → New Notebook → File → Import Notebook** and upload it.
3. In the right panel, under **Session options**, set:
   - **Accelerator:** GPU T4 x2 (or P100);
   - **Internet:** On. This needs a phone-verified account.
4. Edit `JOBS` and `STEPS` in the first cell if you want to change what runs.
5. Click **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the tab.
6. When it finishes, open the version, go to **Output**, and download `results.zip`, which holds the summary,
   plots, videos and logs.

Kaggle stops a session at 12 h, and the notebook ends training before then. To continue, use **Add Input →
Your Work →** the previous version's output, then commit again. The notebook copies over `runs/` (with
checkpoints every 10k steps) and resumes.

## RunPod (pay per second, any GPU)

1. Sign up at runpod.io and add credit under **Billing**.
2. Go to **Pods → Deploy** and pick a GPU. Any recent one is plenty, since the networks are small: RTX 3090,
   RTX 4090 or RTX A4000/A5000 on *Community Cloud* are the cheapest. Prefer a pod with **at least 8 vCPUs**,
   because each run also uses a CPU core for HER sampling and the simulator.
   - **Template:** RunPod PyTorch (2.x). It has CUDA torch, JupyterLab and a web terminal.
   - **Container disk:** 20 GB.
   - **Volume:** 20 GB, mounted at `/workspace`. Only `/workspace` survives stopping the pod.
   - Check the hourly price shown before you click **Deploy**.
3. Once the pod is running, click **Connect → Start Web Terminal** (or open JupyterLab → Terminal) and run:

   ```bash
   cd /workspace
   git clone https://github.com/GugulothBhuvan/ur5e-reach-ddpg-her
   cd ur5e-reach-ddpg-her
   apt-get update && apt-get install -y libosmesa6
   pip install stable-baselines3==2.9.2 sb3-contrib==2.9.2 gymnasium==1.4.0 panda-gym==3.0.7 \
       mujoco==3.17.0 imageio imageio-ffmpeg pyyaml pandas matplotlib pytest
   python -m pytest -q tests
   nohup python run_all.py --jobs panda_ddpg:0,1,2 ur5e_ddpg:0 --steps 100000 --parallel 4 > run_all.log 2>&1 &
   tail -f run_all.log        # Ctrl-C stops watching; training keeps running
   ```

   Do **not** reinstall torch; the template's CUDA build is the one you want. With more vCPUs you can raise
   `--parallel` and add jobs, for example `ur5e_ddpg:0,1,2`.
4. Check progress with `tail logs/*.log`. `nvidia-smi` shows GPU use.
5. To get the results:
   ```bash
   zip -r results.zip results
   ```
   Then download `results.zip` from the JupyterLab file browser (right-click → Download). Alternatively,
   commit `results/` and push it to GitHub with a personal access token.
6. **Stop the pod as soon as you're done.** A running pod is billed per second.
   - **Stop** keeps `/workspace`, with small storage charges.
   - **Terminate** deletes everything and stops all charges.
   If the pod is stopped mid-run, start it again and rerun the same `run_all.py` command; it resumes from the
   last checkpoint.

## Same thing anywhere else (AWS, Krutrim, Azure, the institute HPC, ...)

Any Linux machine with an NVIDIA GPU works the same way as the RunPod steps:
1. Start from an image that already has CUDA PyTorch.
2. Clone the repo.
3. Install `libosmesa6` and the pip packages above.
4. Run `run_all.py` under `nohup`, `tmux` or the cluster's batch scheduler.
