import type { Transport, CallToolRequest, CallToolResult } from './types.js';
import WebSocket from 'ws';
import * as readline from 'readline';
import { z } from 'zod';
import { StructuredLogger } from './logging.js';

// Define request schema here since it's transport-specific
const CallToolRequestSchema = z.object({
  name: z.string(),
  parameters: z.record(z.any())
});

export class WebSocketTransport implements Transport {
  private ws: WebSocket;
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;
  private logger = StructuredLogger.getInstance();

  constructor(url: string) {
    this.ws = new WebSocket(url);
  }

  public async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = async (event) => {
        if (!this.requestHandler) { return; }
        try {
          const rawRequest = JSON.parse(event.data.toString());
          const request = CallToolRequestSchema.parse(rawRequest); // Validate
          const response = await this.requestHandler(request);
          this.ws.send(JSON.stringify(response));
        } catch (error) {
          // Send validation errors
          this.ws.send(JSON.stringify({
            success: false,
            error: error instanceof Error ? error.message : 'Invalid request'
          }));
        }
      };
    });
  }

  public async disconnect(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.ws.close();
      resolve();
    });
  }

  public onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }
}

export class StdioTransport implements Transport {
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;
  private readline: readline.Interface;
  private logger = StructuredLogger.getInstance();

  constructor() {
    this.readline = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  async connect(): Promise<void> {
    this.readline.on('line', async (line: string) => {
      if (!this.requestHandler) {
        this.logger.error('transport', 'No request handler registered');
        return;
      }

      try {
        const request = JSON.parse(line) as CallToolRequest;
        const response = await this.requestHandler(request);
        this.logger.info('transport', 'Response', { response });
        process.stdout.write(`${JSON.stringify(response)}\n`);
      } catch (error) {
        this.logger.error('transport', 'Error handling input', error as Error);
        process.stdout.write(`${JSON.stringify({
          id: null,
          error: {
            message: error instanceof Error ? error.message : String(error)
          }
        })}\n`);
      }
    });

    return Promise.resolve();
  }

  async disconnect(): Promise<void> {
    this.readline.close();
    return Promise.resolve();
  }

  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }

} 