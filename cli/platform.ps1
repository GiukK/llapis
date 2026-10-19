# ===========================================================================
# cli/platform.ps1 - the OS-specific pieces of the CLI, kept in one place (Windows today).
# ===========================================================================
# Everything here has a one-call equivalent elsewhere (Linux: prctl(PR_SET_PDEATHSIG) /
# a process group), so a port replaces this file, not the callers.

# Tie a child process tree to THIS PowerShell process: the child goes into a Job Object with
# KILL_ON_JOB_CLOSE whose only handle lives here, and every process it spawns later (the model
# child) inherits the job. When this shell dies for ANY reason -- Ctrl+C, the terminal's X, a
# kill by PID -- Windows closes the handle and kills the whole tree. Without it, killing the
# launcher left router + model child alive holding ~14 GiB of commit until `llapis stop`
# (findings 189). Best effort: on failure it says so and the old behaviour (finally + stop) stays.
function Register-LlapisChildTree([System.Diagnostics.Process]$Proc) {
  try {
    if (-not ('Llapis.KillOnClose' -as [type])) {
      Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace Llapis {
  public static class KillOnClose {
    [StructLayout(LayoutKind.Sequential)] struct BASIC { public long a, b; public uint LimitFlags; public UIntPtr c, d; public uint e; public UIntPtr f; public uint g, h; }
    [StructLayout(LayoutKind.Sequential)] struct IOC { public ulong a, b, c, d, e, f; }
    [StructLayout(LayoutKind.Sequential)] struct EXT { public BASIC Basic; public IOC Io; public UIntPtr p, q, r, s; }
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr CreateJobObject(IntPtr sa, string name);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetInformationJobObject(IntPtr job, int cls, ref EXT info, uint len);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr proc);
    // returns the job handle (keep it alive: closing it kills the tree), or IntPtr.Zero + throws
    public static IntPtr Attach(IntPtr process) {
      IntPtr job = CreateJobObject(IntPtr.Zero, null);
      if (job == IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
      EXT info = new EXT(); info.Basic.LimitFlags = 0x2000;   // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
      if (!SetInformationJobObject(job, 9, ref info, (uint) Marshal.SizeOf(typeof(EXT)))) throw new System.ComponentModel.Win32Exception();
      if (!AssignProcessToJobObject(job, process)) throw new System.ComponentModel.Win32Exception();
      return job;
    }
  }
}
'@
    }
    $script:LlapisChildJob = [Llapis.KillOnClose]::Attach($Proc.Handle)   # never closed: the OS does it at our exit
    return $true
  } catch {
    Write-Host "[llapis] note: could not tie the server to this window ($($_.Exception.Message)); if you close it, run 'llapis stop'." -ForegroundColor DarkYellow
    return $false
  }
}
