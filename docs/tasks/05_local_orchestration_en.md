# Single-workstation orchestration

[Dispatch diagram](local_orchestration_en.pdf) · [Editable LaTeX](local_orchestration_en.tex) · [中文](05_local_orchestration_zh.md)

## Start the console on the training host

**1. In each new terminal, enter the directory containing `main.py`.** On this host the entry point is under `deeprl_signal_control/cbwce_standalone`. An error ending in `deeprl_signal_control/main.py` means you are one directory too high. Use the training environment's Python and set the `CBWCE_RUNTIME_LOCK`, data and cache paths for your experiment.

~~~bash
cd "$HOME/Journal/deeprl_signal_control/cbwce_standalone"
export CBWCE_PYTHON="${CBWCE_PYTHON:-python3}"
export CBWCE_FLOW="$HOME/cbwce-workflows/experiment-02"
~~~

**2. Initialize each new workflow once.** Choose a directory outside the project that does not yet exist. The example uses formal validation design seeds 73001 for Grid and 73002 for Monaco. Development uses 9001. Initialization freezes the manifests. If `$CBWCE_FLOW/workflow.json` already exists, go straight to step 3.

~~~bash
"$CBWCE_PYTHON" main.py local init-workflow --workflow "$CBWCE_FLOW" \
  --design-seed-grid 73001 --design-seed-monaco 73002
~~~

**3. Start the console.** On this workstation the storage progress page uses port 8765, so the training console examples use 8766. If the console is already running for this workflow, open its URL directly. To restart the service later, repeat steps 1 and 3 with the same training environment and workflow directory. To keep dispatch running after logging out of SSH, run the console in a `tmux` session on the training host.

~~~bash
"$CBWCE_PYTHON" main.py local dashboard --workflow "$CBWCE_FLOW" --port 8766
~~~

Once the terminal prints the address, open the [console](http://127.0.0.1:8766) in a browser on the training host. A new workflow opens in manual mode with 24 tuning candidates. Choose **Run automatically** to start with the parameter inventory, or use **Run stage** and **Run selected**. Every child runs with the same interpreter. Formal verification requires the locked runtime and the full extended gate.

## Connect from another computer

Leave the console running on the training host. In a terminal **on your own computer**, run the SSH command below. The example uses this workstation's account and hostname. If you normally connect using an IP address or SSH alias, use your usual login target as the last argument. The SSH server must allow TCP port forwarding.

~~~bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8766:127.0.0.1:8766 sdc_joran@sdc-1000060-2110363
~~~

Keep the tunnel terminal open and open the [console](http://127.0.0.1:8766) **in your own browser** to view curves, pause jobs or resume them. On a later visit, reconnect the tunnel if the host service is still running. The console checks the request port. To use another free port, change the dashboard port, both ports in the SSH command and the browser address together.

## Workflow stages

|Stage|Release condition|Output|
|---|---|---|
|Inventory and tuning|Inspect six map/controller combinations, then run four candidates per combination for 1,000 episodes each|Dimensions, parameter counts, validation scores|
|Freeze and verify|Select the six winners, place their INIs in an isolated source copy, then verify that copy|Frozen configuration set, hashes and gate. Candidate checkpoints are retired under the existing verified cleanup policy|
|Baseline 01–06|Gate passes|Six best parent controllers|
|WCE 07–12 and continuation 13–30|Each eligible parent independently releases one WCE, baseline continuation, random group and domain randomization|24 downstream jobs across six parents|
|Fixed 31–36 and online 37–42|Each eligible offline WCE releases two continuations with its own parent|12 downstream jobs|
|Evaluate and report|Each completed continuation releases 23 scenario jobs for its best controller through stopping|690 jobs, 10 paired rollouts each, then the complete comparison report|

## Resource queue

The development queue admits up to 12 of its 24 candidates from the first poll. Candidate rank interleaves the two maps and three algorithms. Each worker reserves one CPU slot and uses one thread per computation library. With sufficient resources, a finished job immediately makes room for the next candidate. CPU utilization varies during simulation, validation and writes.

Formal jobs share a 24-worker ceiling and the available CPU budget (16 of 20 cores on this host). With sufficient RAM and disk, 24 ready one-core jobs give 16 running and eight waiting. Each eligible baseline releases its WCE and three direct continuations. Each eligible WCE releases Fixed and Online WCE continuations. Each completed continuation releases its evaluations.

Every five seconds the queue scans all ready jobs and admits those that fit. Each worker initially reserves a 2 GiB RSS estimate and 1 GiB of checkpoint headroom. Observed process-tree peaks can raise these estimates. Admission budgets 1.3 times RSS, leaves 20% of RAM available for the system and reserves 10 GiB of disk. DQL full-replay measurements proceed alongside other workers.

## Operator controls

**Run stage** and **Run selected** authorize one stage or a selection of jobs in manual mode. Running work finishes normally. Selecting **Development tuning → Run stage** also authorizes the model inventory and the selection and verification of winners. The workflow cards show tuning, baseline, offline WCE, the five continuations and evaluation. Expand the checks panel for configuration checks and the final report. The task detail view provides dependencies, a validation curve, the latest log and recorded results. The Results section displays paired outputs after the complete testing matrix.

**Pause all** holds dispatch and requests durable pauses for active jobs. Individual tasks can also be paused. Training saves at an episode boundary; evaluation keeps completed rollout shards. Inventory, Freeze, Verify and Report finish their active step before the next one pauses. **Resume queue** clears a global hold. **Resume selected** clears individual holds. **Retry failed** retries eligible failed tasks. A phase error has its own retry button. Parent or WCE review decisions still apply.

Closing the browser leaves the scheduler running. Restart the console with the same interpreter and directory to reattach to the phase and workers. Keep the dashboard command running, for example in a terminal multiplexer, for automatic transitions between phases. Persistent state, measurements, attempts, logs and reports live in the workflow directory.

## Live curves

Select a task to inspect its curves. The page refreshes every five seconds. Block mean queues update after each 600-second demand block. Each complete episode adds one training point. Fixed validation measures mean queue over the entire 6600 seconds at episode zero and every 100 completed episodes, using realization 0 once. The circled validation point selects the lowest queue for a controller or highest queue for offline WCE.

Reward panels default to the scaled, clipped immediate reward used by the learner. Select raw reward to inspect environment units. Controller points average equally over five-second decisions and junctions. MA2C includes neighbour queues; PPO and DQL use the global queue reward. WCE averages eleven block rewards. Its raw reward is mean queue. Online WCE displays both training roles and validates the controller against external demand. Fixed WCE displays controller rewards. Validation computes the same reward transformation with frozen parameters.

Use block mean queue (vehicles) to follow congestion during an episode and fixed validation to judge progress. Completed-trip waiting time (seconds) and completion counts are available from final-test TripInfo. Per-second queue samples are accumulated online. Compact reward summaries accompany existing episode and validation records; recovery removes superseded points from the displayed curves.
