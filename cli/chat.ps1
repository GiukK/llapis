# ===========================================================================
# llapis/chat.ps1 - server-backed streaming chat REPL for llapis.ps1.
# ===========================================================================
# DOT-SOURCED by llapis.ps1. Invoke-LlapisChatRepl is a self-contained SSE
# client over the local OpenAI-compatible API (no script-scope deps); the serve
# path calls it once the model is loaded. Returns "exit" or "change".
# ===========================================================================

# C2 chat client: a thin streaming REPL over the local OpenAI-compatible API. Returns 'exit'
# or 'change'.
# SSE via HttpWebRequest (PS 5.1 Invoke-RestMethod cannot stream); the final chunk carries
# `timings` (server populates it on stop) -> per-turn tok/s + MTP acceptance.
function Invoke-LlapisChatRepl {
  param([string]$Base, [string]$ModelId)
  $messages = New-Object System.Collections.ArrayList
  try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch {}
  Write-Host ""
  Write-Host "Chat ready. Commands: /new (clear history), /change (setup), /exit" -ForegroundColor DarkGray
  Write-Host ""
  while ($true) {
    Write-Host -NoNewline "> "
    $user = [Console]::In.ReadLine()
    if ($null -eq $user) { return 'exit' }          # EOF / non-interactive stdin closed
    $user = "$user".Trim()
    if (-not $user) { continue }
    if ($user -in @('/exit', '/quit', '/bye')) { return 'exit' }
    if ($user -eq '/change') { return 'change' }
    if ($user -eq '/new') { $messages.Clear(); Write-Host "(history cleared)" -ForegroundColor DarkGray; continue }
    # Variable-KV S6a dev command: schedule a hot-slab rebuild at a new uniform K (applied by the
    # runtime at the next turn end). Deliberately undocumented in the banner (gate/dev use;
    # the Phase-3 rebalance policy will drive this automatically).
    if ($user -match '^/rebuildk\s+(\d+)$') {
      try {
        $null = Invoke-RestMethod -Method Post -Uri "$Base/moe/rebuild_k?k=$($Matches[1])" -TimeoutSec 30
        Write-Host ("[rebuildk] K={0} scheduled (applies at the next turn end)" -f $Matches[1]) -ForegroundColor DarkGray
      } catch {
        Write-Host ("[rebuildk] FAILED: {0}" -f $_.Exception.Message) -ForegroundColor Red
      }
      continue
    }
    # Variable-KV S6b dev command: rebuild the context at a new n_ctx (KV replayed). Applied by
    # the runtime at the next all-idle quiescent point (i.e. right away between turns once the
    # queue wakes, or at the end of the next turn).
    if ($user -match '^/rebuildctx\s+(\d+)$') {
      try {
        $null = Invoke-RestMethod -Method Post -Uri "$Base/moe/rebuild_ctx?n=$($Matches[1])" -TimeoutSec 30
        Write-Host ("[rebuildctx] n_ctx={0} scheduled" -f $Matches[1]) -ForegroundColor DarkGray
      } catch {
        Write-Host ("[rebuildctx] FAILED: {0}" -f $_.Exception.Message) -ForegroundColor Red
      }
      continue
    }
    [void]$messages.Add(@{ role = 'user'; content = $user })

    $json  = ConvertTo-Json @{ model = $ModelId; stream = $true; messages = @($messages) } -Depth 8 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $assistant = New-Object System.Text.StringBuilder
    $timings = $null
    try {
      $req = [System.Net.HttpWebRequest]::Create("$Base/v1/chat/completions")
      $req.Method = 'POST'; $req.ContentType = 'application/json'
      $req.Timeout = 600000; $req.ReadWriteTimeout = 600000
      $req.ContentLength = $bytes.Length
      $rs = $req.GetRequestStream(); $rs.Write($bytes, 0, $bytes.Length); $rs.Close()
      $resp   = $req.GetResponse()
      $reader = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
      while ($null -ne ($line = $reader.ReadLine())) {
        if (-not $line.StartsWith('data: ')) { continue }
        $payload = $line.Substring(6)
        if ($payload -eq '[DONE]') { break }
        try { $chunk = $payload | ConvertFrom-Json } catch { continue }
        $dReason = $chunk.choices[0].delta.reasoning_content
        if ($dReason) { Write-Host -NoNewline $dReason -ForegroundColor DarkGray }   # model thinking, dimmed
        $delta = $chunk.choices[0].delta.content
        if ($delta) { Write-Host -NoNewline $delta; [void]$assistant.Append($delta) }
        if ($chunk.timings) { $timings = $chunk.timings }
      }
      $reader.Close(); $resp.Close()
    } catch {
      Write-Host ""
      Write-Host ("[chat] request failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
      $messages.RemoveAt($messages.Count - 1)      # drop the failed turn, keep history consistent
      continue
    }
    Write-Host ""
    if ($timings) {
      $stat = "  [{0} tok/s" -f [math]::Round([double]$timings.predicted_per_second, 1)
      if ($timings.draft_n -gt 0) {
        $stat += (", MTP accept {0}%" -f [math]::Round(100.0 * $timings.draft_n_accepted / $timings.draft_n, 0))
      }
      Write-Host ($stat + "]") -ForegroundColor DarkGray
    }
    [void]$messages.Add(@{ role = 'assistant'; content = $assistant.ToString() })
    Write-Host ""
  }
}
