# Environmental Observation Agent

This repository accompanies a case study of how language models use tools to answer questions about daily weather observations. It uses temperature and precipitation records for a weather station in Seoul (KSM00047108) from the [NOAA GHCN-Daily dataset](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily). The 53 cases cover data selection, aggregation, missing values, source conflicts, and tool errors.

## Try the observation tool

From the repository root, run the example with Python 3. It uses the standard library and runs without a language model.

```bash
python3 example.py
```

The example finds days from August 13 to 27, 2024 with a mean temperature of at least 29 °C and precipitation of at most 3 mm. The expected result is:

```json
{
  "status": "ok",
  "dates": ["2024-08-13", "2024-08-16", "2024-08-18", "2024-08-19"]
}
```

## Code and inputs

| File | Purpose |
|---|---|
| [runtime.py](work/case_study_v5/runtime.py) | Queries observations, calculates aggregates, and handles dataset state and tool errors. |
| [main_adapter_v1.py](work/case_study_v5/main_adapter_v1.py) | Connects the agent workflow to a local Ollama service and records model requests and responses. |
| [run_main_v1.py](work/case_study_v5/run_main_v1.py) | Runs the cases in a fixed order and enforces time and tool-call limits. |
| [tasks.json](outputs/case_study_v5/tasks.json) | Defines the 53 questions, follow-up turns, reference answers, and assessment criteria. |
| [packets.json](outputs/case_study_v5/packets.json) | Defines the 11 data packets, including their observations and handling rules. |
| [execution_settings_v2.json](outputs/case_study_v5/execution_settings_v2.json) | Records model identifiers, generation settings, and execution limits. |
| [design.json](outputs/case_study_v5/design.json) | Provides the system prompt used to initialize the agent session. |
| [scoring_ledger_template.json](outputs/case_study_v5/scoring_ledger_template.json) | Defines the fields used to record assessments for each case. |

## Full experiment and result verification

The study used four local models on the same 53 cases, for 212 model–case runs. Running the full experiment requires Ollama 0.33.3, the model versions recorded in the execution settings, and the frozen inputs and preflight manifests documented in Appendix A.2 of the thesis. That appendix also points to the reference calculations, execution records, and saved scores used to check the reported results.
