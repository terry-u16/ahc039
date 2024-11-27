import json
import math
import os
import subprocess

import optuna


# TODO: Write parameter suggestions here
def generate_params(trial: optuna.trial.Trial) -> dict[str, str]:
    # for more information, see https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    params = {
        "AHC_TEMP0": str(trial.suggest_float("temp0", 5e1, 1e3, log=True)),
        "AHC_TEMP1": str(trial.suggest_float("temp1", 1e1, 1e2, log=True)),
        "AHC_TEMP2": str(trial.suggest_float("temp2", 3e0, 3e1, log=True)),
        "AHC_TEMP3": str(trial.suggest_float("temp3", 1e0, 1e1, log=True)),
        "AHC_TEMP4": str(trial.suggest_float("temp4", 5e-1, 1e1, log=True)),
        "AHC_TEMP5": str(trial.suggest_float("temp5", 3e-1, 3e0, log=True)),
        "AHC_TEMP6": str(trial.suggest_float("temp6", 1e-1, 1e0, log=True)),
        "AHC_DURATION0": str(trial.suggest_float("duration0", 1e0, 1e2, log=True)),
        "AHC_DURATION1": str(trial.suggest_float("duration1", 1e0, 1e2, log=True)),
        "AHC_DURATION2": str(trial.suggest_float("duration2", 1e0, 1e2, log=True)),
        "AHC_DURATION3": str(trial.suggest_float("duration3", 1e0, 1e2, log=True)),
        "AHC_DURATION4": str(trial.suggest_float("duration4", 1e0, 1e2, log=True)),
        "AHC_DURATION5": str(trial.suggest_float("duration5", 1e0, 1e2, log=True)),
        "AHC_PARALLEL0": str(trial.suggest_int("parallel0", 1, 50)),
        "AHC_PARALLEL1": str(trial.suggest_int("parallel1", 1, 20)),
        "AHC_PARALLEL2": str(trial.suggest_int("parallel2", 1, 10)),
        "AHC_PARALLEL3": str(trial.suggest_int("parallel3", 1, 5)),
        "AHC_PARALLEL4": str(trial.suggest_int("parallel4", 1, 3)),
        "AHC_PARALLEL5": str(trial.suggest_int("parallel5", 1, 2)),
    }

    return params


# TODO: Customize the score extraction code here
def extract_score(result: dict[str, str]) -> float:
    absolute_score = result["score"]  # noqa: F841
    log10_score = math.log10(absolute_score) if absolute_score > 0.0 else 0.0  # noqa: F841
    relative_score = result["relative_score"]  # noqa: F841

    score = absolute_score  # for absolute score problems
    # score = log10_score       # for relative score problems (alternative)
    # score = relative_score    # for relative score problems

    return score


# TODO: Set the direction to minimize or maximize
def get_direction() -> str:
    # direction = "minimize"
    direction = "maximize"
    return direction


# TODO: Set the timeout (seconds) or the number of trials
def run_optimization(study: optuna.study.Study) -> None:
    study.optimize(Objective(), timeout=3600)
    # study.optimize(Objective(), n_trials=100)


class Objective:
    def __init__(self) -> None:
        pass

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = generate_params(trial)
        env = os.environ.copy()
        env.update(params)

        scores = []

        cmd = [
            "pahcer",
            "run",
            "--json",
            "--shuffle",
            "--no-result-file",
            "--freeze-best-scores",
        ]

        if trial.number != 0:
            cmd.append("--no-compile")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env,
        )

        # see also: https://tech.preferred.jp/ja/blog/wilcoxonpruner/
        for line in process.stdout:
            result = json.loads(line)

            # If an error occurs, stop the process and raise an exception
            if result["error_message"] != "":
                process.send_signal(subprocess.signal.SIGINT)
                raise RuntimeError(result["error_message"])

            score = extract_score(result)
            seed = result["seed"]
            scores.append(score)
            trial.report(score, seed)

            if trial.should_prune():
                print(f"Trial {trial.number} pruned.")
                process.send_signal(subprocess.signal.SIGINT)

                objective_value = sum(scores) / len(scores)
                is_better_than_best = (
                    trial.study.direction == optuna.study.StudyDirection.MINIMIZE
                    and objective_value < trial.study.best_value
                ) or (
                    trial.study.direction == optuna.study.StudyDirection.MAXIMIZE
                    and objective_value > trial.study.best_value
                )

                if is_better_than_best:
                    # Avoid updating the best value
                    raise optuna.TrialPruned()
                else:
                    # It is recommended to return the value of the objective function at the current step
                    # instead of raising TrialPruned.
                    # This is a workaround to report the evaluation information of the pruned Trial to Optuna.
                    return sum(scores) / len(scores)

        return sum(scores) / len(scores)


study = optuna.create_study(
    direction=get_direction(),
    study_name="optuna-study",
    pruner=optuna.pruners.WilcoxonPruner(),
    sampler=optuna.samplers.TPESampler(),
)

study.enqueue_trial({
    "temp0": 194.3309382414169,
    "temp1": 46.12704017670174,
    "temp2": 18.227299268925755,
    "temp3": 7.42267951436575,
    "temp4": 0.5060798936653568,
    "temp5": 0.34992860335141246,
    "temp6": 0.10456339020020013,
    "duration0": 79.38382066134952,
    "duration1": 50.67217863285157,
    "duration2": 1.0905098260598414,
    "duration3": 3.6708003135295937,
    "duration4": 2.1089564495993525,
    "duration5": 27.148245362821065,
    "parallel0": 36,
    "parallel1": 4,
    "parallel2": 9,
    "parallel3": 5,
    "parallel4": 3,
    "parallel5": 2,
})

run_optimization(study)

print(f"best params = {study.best_params}")
print(f"best score  = {study.best_value}")
