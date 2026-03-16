import { spawn, type ChildProcess } from "child_process";

export interface ServerConfig {
  llamaServerExe: string;
  modelPath: string;
  serverPort: number;
  gpuLayers: number;
  contextSize: number;
  parallel: number;
  serverUrl?: string | undefined;
}

export class LlamaServerManager {
  private process: ChildProcess | null = null;
  public baseUrl: string;
  private isStopping = false;
  private restartPromise: Promise<void> | null = null;
  private label: string;

  constructor(private config: ServerConfig, label = "LlamaServer") {
    this.baseUrl = config.serverUrl ?? `http://127.0.0.1:${config.serverPort}`;
    this.label = label;
  }

  /** Returns true if using an external server (no process management needed). */
  get isExternal(): boolean {
    return this.config.serverUrl !== undefined;
  }

  async start(): Promise<void> {
    if (this.isExternal) {
      console.log(`[${this.label}] Using external server at ${this.baseUrl}`);
      return;
    }
    if (this.restartPromise) return this.restartPromise;
    if (this.process) return;

    this.isStopping = false;

    const args = [
      "-m", this.config.modelPath,
      "--ctx-size", String(this.config.contextSize),
      "--gpu-layers", String(this.config.gpuLayers),
      "--host", "127.0.0.1",
      "--port", String(this.config.serverPort),
      "--parallel", String(this.config.parallel),
      "--kv-unified",
      "--keep", "-1",
      "-fa", "on",
      "--context-shift",
      "--cache-type-k", "q8_0",
      "--cache-type-v", "q8_0",
    ];

    console.log(`[${this.label}] Starting ${this.config.llamaServerExe} on port ${this.config.serverPort}...`);

    this.process = spawn(this.config.llamaServerExe, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    return new Promise((resolve, reject) => {
      let isReady = false;

      const checkReady = (data: Buffer) => {
        const text = data.toString();
        process.stderr.write(text);

        if (
          text.includes("HTTP server listening") ||
          text.includes("server listening") ||
          text.includes("starting the main loop") ||
          text.includes("update_slots: all slots are idle")
        ) {
          if (!isReady) {
            isReady = true;
            this.process?.stdout?.off("data", checkReady);
            this.process?.stderr?.off("data", checkReady);
            this.process?.stderr?.on("data", (d: Buffer) => process.stderr.write(d));
            console.log(`\n[${this.label}] Ready at ${this.baseUrl}`);
            resolve();
          }
        }
      };

      this.process!.stdout?.on("data", checkReady);
      this.process!.stderr?.on("data", checkReady);

      this.process!.on("error", (err) => {
        if (!isReady) reject(err);
        this.process = null;
      });

      this.process!.on("exit", (code) => {
        this.process = null;
        if (!isReady) {
          reject(new Error(`${this.label} exited early with code ${code}`));
        } else if (!this.isStopping) {
          console.error(`\n[${this.label}] CRASH DETECTED! Exited with code ${code}. Auto-restarting...`);
          this.restartPromise = this.start().then(() => {
            console.log(`[${this.label}] Auto-restart recovered.`);
            this.restartPromise = null;
          }).catch(e => {
            console.error(`[${this.label}] Auto-restart failed:`, e);
            this.restartPromise = null;
          });
        }
      });

      setTimeout(() => {
        if (!isReady) {
          reject(new Error(`Timeout waiting for ${this.label} to be ready.`));
          this.stop();
        }
      }, 60000);
    });
  }

  async stop(): Promise<void> {
    if (this.isExternal) return;
    this.isStopping = true;
    const proc = this.process;
    if (!proc) return;

    console.log(`[${this.label}] Stopping...`);

    return new Promise((resolve) => {
      const killTimer = setTimeout(() => {
        console.log(`[${this.label}] SIGTERM timeout, using SIGKILL...`);
        try { proc.kill("SIGKILL"); } catch {}
        this.process = null;
        resolve();
      }, 5000);

      proc.once("exit", () => {
        clearTimeout(killTimer);
        this.process = null;
        console.log(`[${this.label}] Stopped.`);
        resolve();
      });

      proc.kill("SIGTERM");
    });
  }

  async forceRestart(): Promise<void> {
    if (this.isExternal) return;
    console.warn(`\n[${this.label}] FORCE RESTART INITIATED`);
    this.isStopping = true;
    if (this.process) {
      try { this.process.kill("SIGKILL"); } catch {}
      this.process = null;
    }
    await new Promise(r => setTimeout(r, 1000));
    return this.start();
  }
}
