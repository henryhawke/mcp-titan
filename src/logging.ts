import * as fs from 'fs/promises';
import * as path from 'path';

export enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
}

export interface LogEntry {
    timestamp: string;
    level: string;
    operation: string;
    message: string;
    metadata?: Record<string, any>;
    error?: {
        name: string;
        message: string;
        stack?: string;
    };
}

export class StructuredLogger {
    private static instance: StructuredLogger;
    private logBuffer: LogEntry[] = [];
    private flushInterval?: NodeJS.Timeout;
    private logLevel: LogLevel = LogLevel.INFO;
    private logDir: string;
    private maxFileSize = 10 * 1024 * 1024; // 10MB
    private maxFiles = 5;

    private constructor(logDir: string) {
        this.logDir = logDir;
        this.startFlushInterval();
    }

    public static getInstance(logDir?: string): StructuredLogger {
        if (!StructuredLogger.instance) {
            StructuredLogger.instance = new StructuredLogger(
                logDir || path.join(process.cwd(), '.titan_memory', 'logs')
            );
        }
        return StructuredLogger.instance;
    }

    public setLogLevel(level: LogLevel): void {
        this.logLevel = level;
    }

    public debug(operation: string, message: string, metadata?: Record<string, any>): void {
        if (this.logLevel <= LogLevel.DEBUG) {
            this.log('DEBUG', operation, message, metadata);
        }
    }

    public info(operation: string, message: string, metadata?: Record<string, any>): void {
        if (this.logLevel <= LogLevel.INFO) {
            this.log('INFO', operation, message, metadata);
        }
    }

    public warn(operation: string, message: string, metadata?: Record<string, any>): void {
        if (this.logLevel <= LogLevel.WARN) {
            this.log('WARN', operation, message, metadata);
        }
    }

    public error(operation: string, message: string, error?: Error, metadata?: Record<string, any>): void {
        if (this.logLevel <= LogLevel.ERROR) {
            const errorData = error ? {
                name: error.name,
                message: error.message,
                stack: error.stack
            } : undefined;

            this.log('ERROR', operation, message, metadata, errorData);
        }
    }

    private log(
        level: string,
        operation: string,
        message: string,
        metadata?: Record<string, any>,
        error?: { name: string; message: string; stack?: string }
    ): void {
        const safeMetadata = this.redactMetadata(metadata);
        const entry: LogEntry = {
            timestamp: new Date().toISOString(),
            level,
            operation,
            message,
            metadata: safeMetadata,
            error
        };

        const consoleMsg = `[${entry.timestamp}] ${level} [${operation}]: ${message}`;
        switch (level) {
            case 'ERROR':
                console.error(consoleMsg, safeMetadata, error);
                break;
            case 'WARN':
                console.warn(consoleMsg, safeMetadata);
                break;
            case 'DEBUG':
                console.debug(consoleMsg, safeMetadata);
                break;
            default:
                console.log(consoleMsg, safeMetadata);
        }

        this.logBuffer.push(entry);

        if (this.logBuffer.length >= 100) {
            this.flush().catch(err => console.error('Failed to flush logs:', err));
        }
    }

    private startFlushInterval(): void {
        this.flushInterval = setInterval(() => {
            this.flush().catch(err => console.error('Failed to flush logs:', err));
        }, 10000);
    }

    public async flush(): Promise<void> {
        if (this.logBuffer.length === 0) {
            return;
        }

        try {
            await fs.mkdir(this.logDir, { recursive: true });

            const today = new Date().toISOString().split('T')[0];
            const logFile = path.join(this.logDir, `titan-${today}.log`);

            await this.rotateLogsIfNeeded(logFile);

            const logLines = `${this.logBuffer.map(entry => JSON.stringify(entry)).join('\n')}\n`;
            await fs.appendFile(logFile, logLines, 'utf-8');

            this.logBuffer = [];
        } catch (error) {
            console.error('Failed to write logs:', error);
        }
    }

    private async rotateLogsIfNeeded(logFile: string): Promise<void> {
        try {
            const stats = await fs.stat(logFile);

            if (stats.size >= this.maxFileSize) {
                for (let i = this.maxFiles - 1; i > 0; i--) {
                    const oldFile = logFile.replace('.log', `.${i}.log`);
                    const newFile = logFile.replace('.log', `.${i + 1}.log`);
                    try {
                        await fs.rename(oldFile, newFile);
                    } catch {
                        // ignore missing files
                    }
                }

                await fs.rename(logFile, logFile.replace('.log', '.1.log'));
            }
        } catch (error: any) {
            if (error.code !== 'ENOENT') {
                console.error('Failed to rotate logs:', error);
            }
        }
    }

    public async dispose(): Promise<void> {
        if (this.flushInterval) {
            clearInterval(this.flushInterval);
        }
        await this.flush();
    }

    private redactMetadata(metadata?: Record<string, any>): Record<string, any> | undefined {
        if (!metadata) { return metadata; }
        const redacted: Record<string, any> = {};
        const sensitiveKeys = ['token', 'password', 'secret', 'authorization', 'auth', 'key'];

        for (const [key, value] of Object.entries(metadata)) {
            if (sensitiveKeys.some(s => key.toLowerCase().includes(s))) {
                redacted[key] = '[REDACTED]';
            } else {
                redacted[key] = value;
            }
        }
        return redacted;
    }
}
