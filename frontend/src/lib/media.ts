export type MicState =
  | "checking"
  | "unsupported"
  | "insecure"
  | "permission-denied"
  | "no-device"
  | "ready"
  | "recording"
  | "stopped"
  | "failed";

export async function checkMicrophone(requestAccess = false): Promise<MicState> {
  const local = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  if (!window.isSecureContext && !local) return "insecure";
  if (!navigator.mediaDevices?.getUserMedia) return "unsupported";
  try {
    if (requestAccess) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.some((device) => device.kind === "audioinput") ? "ready" : "no-device";
  } catch (error: any) {
    if (["NotAllowedError", "SecurityError", "PermissionDeniedError"].includes(error?.name)) {
      return "permission-denied";
    }
    return "failed";
  }
}

