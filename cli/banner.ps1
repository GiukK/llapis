# ===========================================================================
# llapis/banner.ps1 - startup banner: the LLapis logo mark + LED-style wordmark.
# ===========================================================================
# Dot-sourced by llapis.ps1 alongside the other llapis/*.ps1 modules; defines
# Show-LlapisBanner, called from server.ps1 right before the chat/serve header
# (both product entry points share that one call site). Pure presentation, no
# state, no side effects besides Write-Host.
# ===========================================================================

# Writes one banner row from (text, color) tuples; $null color = no -ForegroundColor (blank fill).
function Write-BannerRow {
  param([Parameter(Mandatory)][array]$Segments)
  foreach ($seg in $Segments) {
    if ($seg[1]) { Write-Host -NoNewline -Object $seg[0] -ForegroundColor $seg[1] }
    else         { Write-Host -NoNewline -Object $seg[0] }
  }
  Write-Host ""
}

function Show-LlapisBanner {
  $blk = [string][char]0x2588   # full block - badge fill / LED segment glyph

  # NOTE: the multiply is done as its own statement inside these helpers, not inline in a tuple
  # literal -- PowerShell's comma operator binds TIGHTER than `*`, so `$blk * 14, 'Color'` parses
  # as `$blk * (14, 'Color')` (measured; see git history for the repro). Building the tuple through
  # a function call sidesteps the ambiguity entirely.
  function Blank([int]$n) { $t = ' ' * $n; ($t, $null) }
  function Fill([int]$n, [string]$c) { $t = $blk * $n; ($t, $c) }

  Write-Host ""
  # ---- logo mark: solid DarkBlue badge, staircased corners approximate the source icon's
  #      chamfered top-right / bottom-left cuts; the two-tall / two-short bar glyph is punched
  #      out as blank holes rather than filled ----
  Write-BannerRow ((Blank 2), (Fill 17 'DarkBlue'), (Blank 8))
  Write-BannerRow ((Blank 2), (Fill 20 'DarkBlue'), (Blank 5))
  Write-BannerRow ((Blank 2), (Fill 23 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 23 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 3 'DarkBlue'), (Blank 2), (Fill 5 'DarkBlue'), (Blank 2), (Fill 11 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 3 'DarkBlue'), (Blank 2), (Fill 5 'DarkBlue'), (Blank 2), (Fill 11 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 3 'DarkBlue'), (Blank 2), (Fill 5 'DarkBlue'), (Blank 2), (Fill 11 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 3 'DarkBlue'), (Blank 2), (Fill 1 'DarkBlue'), (Blank 3), (Fill 1 'DarkBlue'), (Blank 2), (Fill 1 'DarkBlue'), (Blank 3), (Fill 7 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 2), (Fill 23 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 5), (Fill 20 'DarkBlue'), (Blank 2))
  Write-BannerRow ((Blank 8), (Fill 17 'DarkBlue'), (Blank 2))

  # ---- wordmark: LLAPIS in a chunky block pseudo-font (5 cols x 7 rows per letter) ----
  # L's left stroke and A/P's bottom are filled in solid (not strict 7-segment) -- reads
  # better at this block size than an open middle/foot.
  $L = @("$blk    ", "$blk    ", "$blk    ", "$blk    ", "$blk    ", "$blk    ", ($blk * 5))
  $A = @(($blk * 5), "$blk   $blk", "$blk   $blk", ($blk * 5), "$blk   $blk", "$blk   $blk", "$blk   $blk")
  $P = @(($blk * 5), "$blk   $blk", "$blk   $blk", ($blk * 5), "$blk    ", "$blk    ", "$blk    ")
  $I = @(($blk * 5), "  $blk  ", "  $blk  ", "  $blk  ", "  $blk  ", "  $blk  ", ($blk * 5))
  $S = @(($blk * 5), "$blk    ", "$blk    ", ($blk * 5), "    $blk", "    $blk", ($blk * 5))
  $word = @($L, $L, $A, $P, $I, $S)

  Write-Host ""
  for ($r = 0; $r -lt 7; $r++) {
    Write-Host (($word | ForEach-Object { $_[$r] }) -join ' ') -ForegroundColor DarkBlue
  }
  Write-Host ""
}
