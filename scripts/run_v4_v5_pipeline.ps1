param(
  [Parameter(Mandatory=$true)][string]$Candidates,
  [Parameter(Mandatory=$true)][string]$CombinedDir,
  [Parameter(Mandatory=$true)][string]$ProblemMd,
  [string]$OutputRoot = "run_outputs",
  [string]$Model = "deepseek-v4-flash"
)

if (-not $env:DEEPSEEK_API_KEY) {
  throw "Set DEEPSEEK_API_KEY before running."
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$V4 = Join-Path $OutputRoot "agentic_material_corpus_v4"
$V5 = Join-Path $OutputRoot "agentic_material_corpus_v5"

python code/material_pipeline/run_agentic_material_corpus_v4.py `
  --candidates $Candidates `
  --combined-dir $CombinedDir `
  --problem $ProblemMd `
  --output-dir $V4 `
  --model $Model `
  --limit-docs 0 `
  --per-property 12 `
  --max-packets-per-doc 20 `
  --batch-size 6 `
  --workers 4 `
  --timeout-sec 240 `
  --max-tokens 7000

python code/material_pipeline/run_agentic_material_corpus_v5.py `
  --records (Join-Path $V4 "records/submission_candidates.jsonl") `
  --candidates $Candidates `
  --combined-dir $CombinedDir `
  --problem $ProblemMd `
  --output-dir $V5 `
  --model $Model `
  --workers 4 `
  --records-per-call 5 `
  --timeout-sec 240 `
  --max-tokens 6000 `
  --exclude-review-sources
