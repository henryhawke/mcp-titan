import { EventEmitter } from 'events';
import { GitHubWorkflowManager } from './GitHubWorkflowManager.js';
import { LintingManager } from './LintingManager.js';
import { FeedbackProcessor } from './FeedbackProcessor.js';
import { HopeMemoryModel } from '../model.js';
import type { WorkflowConfig, WorkflowStatus, WorkflowEvent, WorkflowMetrics, HopeMemorySystem } from '../types.js';

export class WorkflowOrchestrator extends EventEmitter {
    private config: WorkflowConfig;
    private memory: HopeMemorySystem;
    private gitHubManager!: GitHubWorkflowManager;
    private lintingManager!: LintingManager;
    private feedbackProcessor!: FeedbackProcessor;
    private status: WorkflowStatus;
    private metrics: WorkflowMetrics;
    private healthCheckInterval?: NodeJS.Timeout;
    private isInitialized = false;

    constructor(config: WorkflowConfig) {
        super();
        this.config = config;
        this.memory = new HopeMemoryModel(config.memory.hopeConfig ?? {});
        this.status = {
            state: 'initializing',
            lastUpdate: new Date(),
            activeWorkflows: [],
            health: 'unknown'
        };
        this.metrics = {
            totalWorkflows: 0,
            successfulWorkflows: 0,
            failedWorkflows: 0,
            averageExecutionTime: 0,
            memoryUsage: 0,
            lastMetricsUpdate: new Date()
        };
    }

    /**
     * Initialize the workflow orchestrator and all components
     */
    async initialize(): Promise<void> {
        try {
            this.emit('status', { type: 'initialization', message: 'Starting workflow orchestrator' });

            // Initialize memory system
            await this.memory.initialize(config.memory.hopeConfig ?? {});
            this.emit('status', { type: 'initialization', message: 'Memory system initialized' });

            // Initialize workflow managers
            this.gitHubManager = new GitHubWorkflowManager(this.config, this.memory);
            this.lintingManager = new LintingManager(this.config, this.memory);
            this.feedbackProcessor = new FeedbackProcessor(this.config, this.memory);

            // Set up event listeners
            this.setupEventListeners();

            // Start health monitoring
            this.startHealthMonitoring();

            this.status.state = 'ready';
            this.status.health = 'healthy';
            this.isInitialized = true;

            this.emit('status', { type: 'ready', message: 'Workflow orchestrator ready' });

        } catch (error) {
            this.status.state = 'error';
            this.status.health = 'unhealthy';
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
    }

    /**
     * Execute a workflow by name with parameters
     */
    async executeWorkflow(workflowName: string, params: Record<string, any> = {}): Promise<any> {
        if (!this.isInitialized) {
            throw new Error('Workflow orchestrator not initialized');
        }

        const workflowId = this.generateWorkflowId();
        const startTime = Date.now();

        try {
            this.status.activeWorkflows.push({
                id: workflowId,
                name: workflowName,
                startTime: new Date(),
                status: 'running'
            });

            this.emit('workflow:start', { id: workflowId, name: workflowName, params });

            let result: any;

            switch (workflowName) {
                case 'auto-release':
                    result = await this.executeAutoRelease(params);
                    break;

                case 'process-issue':
                    result = await this.executeProcessIssue(params);
                    break;

                case 'collect-feedback':
                    result = await this.executeCollectFeedback(params);
                    break;

                case 'quality-check':
                    result = await this.executeQualityCheck(params);
                    break;

                case 'full-pipeline':
                    result = await this.executeFullPipeline(params);
                    break;

                default:
                    throw new Error(`Unknown workflow: ${workflowName}`);
            }

            const executionTime = Date.now() - startTime;
            this.updateMetrics(true, executionTime);
            this.removeActiveWorkflow(workflowId);

            this.emit('workflow:complete', {
                id: workflowId,
                name: workflowName,
                result,
                executionTime
            });

            // Store successful workflow execution in memory
            await this.memory.storeWorkflowMemory('workflow_execution', {
                workflowId,
                workflowName,
                params,
                result,
                executionTime,
                success: true,
                timestamp: new Date()
            });

            return result;

        } catch (error) {
            const executionTime = Date.now() - startTime;
            this.updateMetrics(false, executionTime);
            this.removeActiveWorkflow(workflowId);

            this.emit('workflow:error', {
                id: workflowId,
                name: workflowName,
                error,
                executionTime
            });

            // Store failed workflow execution in memory for learning
            await this.memory.storeWorkflowMemory('workflow_execution', {
                workflowId,
                workflowName,
                params,
                error: error.message,
                executionTime,
                success: false,
                timestamp: new Date()
            });

            throw error;
        }
    }

    /**
     * Execute auto-release workflow
     */
    private async executeAutoRelease(params: any): Promise<any> {
        this.emit('workflow:step', { step: 'auto-release', status: 'starting' });

        // Check if release should be triggered
        const shouldRelease = await this.shouldTriggerRelease(params);
        if (!shouldRelease) {
            return { triggered: false, reason: 'Release conditions not met' };
        }

        // Run pre-release quality checks
        this.emit('workflow:step', { step: 'quality-checks', status: 'running' });
        const qualityResults = await this.lintingManager.runFullQualitySuite();

        if (!qualityResults.passed) {
            throw new Error(`Quality checks failed: ${qualityResults.issues.join(', ')}`);
        }

        // Create release PR
        this.emit('workflow:step', { step: 'create-pr', status: 'running' });
        const releasePR = await this.gitHubManager.createReleasePR();

        this.emit('workflow:step', { step: 'auto-release', status: 'completed' });

        return {
            triggered: true,
            releasePR,
            qualityResults
        };
    }

    /**
     * Execute issue processing workflow
     */
    private async executeProcessIssue(params: { issueNumber: number }): Promise<any> {
        this.emit('workflow:step', { step: 'process-issue', status: 'starting' });

        // Classify and label the issue
        const classification = await this.gitHubManager.processIssue(params.issueNumber);

        // Generate recommendations based on classification
        const recommendations = await this.generateIssueRecommendations(classification);

        // Check for potential automation opportunities
        const automationSuggestions = await this.identifyAutomationOpportunities(classification);

        this.emit('workflow:step', { step: 'process-issue', status: 'completed' });

        return {
            classification,
            recommendations,
            automationSuggestions
        };
    }

    /**
     * Execute feedback collection workflow
     */
    private async executeCollectFeedback(params: any): Promise<any> {
        this.emit('workflow:step', { step: 'collect-feedback', status: 'starting' });

        // Collect feedback from all configured channels
        const feedbackItems = await this.gitHubManager.collectFeedback();

        // Process and analyze feedback
        const analysis = await this.feedbackProcessor.processFeedback(feedbackItems);

        // Generate action items
        const actionItems = await this.generateFeedbackActionItems(analysis);

        this.emit('workflow:step', { step: 'collect-feedback', status: 'completed' });

        return {
            feedbackItems,
            analysis,
            actionItems
        };
    }

    /**
     * Execute quality check workflow
     */
    private async executeQualityCheck(params: { prNumber?: number }): Promise<any> {
        this.emit('workflow:step', { step: 'quality-check', status: 'starting' });

        // Run comprehensive quality checks
        const results = await this.gitHubManager.runQualityChecks(params.prNumber);

        // Generate improvement suggestions
        const suggestions = await this.generateQualityImprovements(results);

        this.emit('workflow:step', { step: 'quality-check', status: 'completed' });

        return {
            ...results,
            suggestions
        };
    }

    /**
     * Execute full pipeline workflow
     */
    private async executeFullPipeline(params: any): Promise<any> {
        this.emit('workflow:step', { step: 'full-pipeline', status: 'starting' });

        const results: Record<string, any> = {};

        // Step 1: Collect and process feedback
        try {
            results.feedback = await this.executeCollectFeedback({});
        } catch (error) {
            console.warn('Feedback collection failed:', error);
            results.feedback = { error: error.message };
        }

        // Step 2: Run quality checks
        try {
            results.quality = await this.executeQualityCheck({});
        } catch (error) {
            console.warn('Quality checks failed:', error);
            results.quality = { error: error.message };
        }

        // Step 3: Check for auto-release
        try {
            results.release = await this.executeAutoRelease({});
        } catch (error) {
            console.warn('Auto-release failed:', error);
            results.release = { error: error.message };
        }

        // Step 4: Generate summary and recommendations
        results.summary = await this.generatePipelineSummary(results);

        this.emit('workflow:step', { step: 'full-pipeline', status: 'completed' });

        return results;
    }

    /**
     * Get current workflow status
     */
    getStatus(): WorkflowStatus {
        return { ...this.status };
    }

    /**
     * Get workflow metrics
     */
    getMetrics(): WorkflowMetrics {
        return { ...this.metrics };
    }

    /**
     * Get workflow history from memory
     */
    async getWorkflowHistory(limit = 10): Promise<any[]> {
        return await this.memory.getWorkflowHistory('workflow_execution', limit);
    }

    /**
     * Handle webhook events
     */
    async handleWebhook(event: string, payload: any): Promise<void> {
        try {
            this.emit('webhook', { event, payload });

            // Delegate to appropriate manager
            await this.gitHubManager.handleWebhook(event, payload);

            // Check if this should trigger any automated workflows
            await this.checkAutomationTriggers(event, payload);

        } catch (error) {
            this.emit('error', { type: 'webhook', event, error });
        }
    }

    /**
     * Shutdown the orchestrator gracefully
     */
    async shutdown(): Promise<void> {
        this.emit('status', { type: 'shutdown', message: 'Shutting down workflow orchestrator' });

        // Stop health monitoring
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        // Wait for active workflows to complete or timeout
        await this.waitForActiveWorkflows(30000); // 30 seconds timeout

        // Shutdown memory system
        await this.memory.shutdown();

        this.status.state = 'stopped';
        this.emit('status', { type: 'stopped', message: 'Workflow orchestrator stopped' });
    }

    // Private methods

    private setupEventListeners(): void {
        // Set up cross-component event handling
        this.on('workflow:error', this.handleWorkflowError.bind(this));
        this.on('memory:error', this.handleMemoryError.bind(this));
    }

    private startHealthMonitoring(): void {
        this.healthCheckInterval = setInterval(async () => {
            await this.performHealthCheck();
        }, 60000); // Check every minute
    }

    private async performHealthCheck(): Promise<void> {
        try {
            // Check memory system health
            const memoryHealth = await this.memory.getHealthStatus();

            // Check workflow manager health
            const workflowHealth = this.checkWorkflowHealth();

            // Update overall health status
            this.status.health = (memoryHealth.healthy && workflowHealth) ? 'healthy' : 'unhealthy';
            this.status.lastUpdate = new Date();

            this.emit('health:check', {
                memory: memoryHealth,
                workflows: workflowHealth,
                overall: this.status.health
            });

        } catch (error) {
            this.status.health = 'unhealthy';
            this.emit('error', { type: 'health-check', error });
        }
    }

    private checkWorkflowHealth(): boolean {
        // Check if we have too many active workflows
        if (this.status.activeWorkflows.length > 10) {
            return false;
        }

        // Check if any workflows have been running too long
        const now = Date.now();
        const staleWorkflows = this.status.activeWorkflows.filter(w =>
            now - w.startTime.getTime() > 300000 // 5 minutes
        );

        return staleWorkflows.length === 0;
    }

    private generateWorkflowId(): string {
        return `wf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private updateMetrics(success: boolean, executionTime: number): void {
        this.metrics.totalWorkflows++;

        if (success) {
            this.metrics.successfulWorkflows++;
        } else {
            this.metrics.failedWorkflows++;
        }

        // Update average execution time
        const totalTime = this.metrics.averageExecutionTime * (this.metrics.totalWorkflows - 1) + executionTime;
        this.metrics.averageExecutionTime = totalTime / this.metrics.totalWorkflows;

        // Update memory usage (approximation)
        this.metrics.memoryUsage = process.memoryUsage().heapUsed;
        this.metrics.lastMetricsUpdate = new Date();
    }

    private removeActiveWorkflow(workflowId: string): void {
        this.status.activeWorkflows = this.status.activeWorkflows.filter(w => w.id !== workflowId);
    }

    private async shouldTriggerRelease(params: any): Promise<boolean> {
        // Check configuration-based conditions
        const config = this.config.features.autoRelease.triggerConditions;

        // Check commit count condition
        if (config.commitCount > 0) {
            // Implementation would check actual commit count
            return true; // Placeholder
        }

        // Check time-based conditions
        if (config.timeThreshold) {
            // Implementation would check time since last release
            return true; // Placeholder
        }

        // Check feature flags
        if (config.featureFlags.length > 0) {
            // Implementation would check feature completion
            return true; // Placeholder
        }

        return false;
    }

    private async generateIssueRecommendations(classification: any): Promise<string[]> {
        const recommendations: string[] = [];

        if (classification.priority === 'critical') {
            recommendations.push('Assign to senior developer immediately');
            recommendations.push('Set up monitoring for this issue type');
        }

        if (classification.complexity === 'complex') {
            recommendations.push('Break down into smaller tasks');
            recommendations.push('Consider architectural review');
        }

        if (classification.estimatedHours > 20) {
            recommendations.push('Consider adding to next sprint planning');
            recommendations.push('May require dedicated milestone');
        }

        return recommendations;
    }

    private async identifyAutomationOpportunities(classification: any): Promise<string[]> {
        const opportunities: string[] = [];

        if (classification.type === 'bug' && classification.component.includes('testing')) {
            opportunities.push('Add automated test case to prevent regression');
        }

        if (classification.type === 'documentation') {
            opportunities.push('Consider auto-generating docs from code comments');
        }

        if (classification.component.includes('api')) {
            opportunities.push('Add API monitoring and alerts');
        }

        return opportunities;
    }

    private async generateFeedbackActionItems(analysis: any): Promise<string[]> {
        const actionItems: string[] = [];

        // Analyze high-priority negative feedback
        const criticalFeedback = analysis.items?.filter((item: any) =>
            item.sentiment === 'negative' && item.priority > 7
        ) || [];

        for (const item of criticalFeedback) {
            actionItems.push(`Address critical feedback: ${item.topics.join(', ')}`);
        }

        // Look for common themes
        const commonTopics = analysis.commonTopics || [];
        for (const topic of commonTopics) {
            if (topic.frequency > 3) {
                actionItems.push(`Investigate recurring topic: ${topic.name}`);
            }
        }

        return actionItems;
    }

    private async generateQualityImprovements(results: any): Promise<string[]> {
        const improvements: string[] = [];

        if (results.results.coverage?.percentage < 80) {
            improvements.push('Increase test coverage to at least 80%');
        }

        if (results.results.security?.vulnerabilities > 0) {
            improvements.push('Address security vulnerabilities before release');
        }

        if (results.results.performance?.issues > 0) {
            improvements.push('Optimize performance bottlenecks');
        }

        return improvements;
    }

    private async generatePipelineSummary(results: any): Promise<any> {
        const summary = {
            timestamp: new Date(),
            status: 'completed',
            steps: Object.keys(results),
            successfulSteps: Object.keys(results).filter(key => !results[key].error),
            failedSteps: Object.keys(results).filter(key => results[key].error),
            recommendations: [] as string[]
        };

        // Generate recommendations based on results
        if (results.quality && !results.quality.error && !results.quality.passed) {
            summary.recommendations.push('Fix quality issues before next release');
        }

        if (results.feedback && !results.feedback.error) {
            summary.recommendations.push('Review and address user feedback');
        }

        if (results.release?.triggered) {
            summary.recommendations.push('Monitor release PR for approvals');
        }

        return summary;
    }

    private async checkAutomationTriggers(event: string, payload: any): Promise<void> {
        // Check if this event should trigger any automated workflows
        const triggers = this.config.integrations.github.webhooks.events;

        if (triggers.includes(event)) {
            switch (event) {
                case 'push':
                    if (payload.ref === `refs/heads/${this.config.repository.branch}`) {
                        await this.executeWorkflow('quality-check', { prNumber: null });
                    }
                    break;

                case 'pull_request.opened':
                    await this.executeWorkflow('quality-check', { prNumber: payload.pull_request.number });
                    break;

                case 'issues.opened':
                    await this.executeWorkflow('process-issue', { issueNumber: payload.issue.number });
                    break;
            }
        }
    }

    private async waitForActiveWorkflows(timeout: number): Promise<void> {
        const startTime = Date.now();

        while (this.status.activeWorkflows.length > 0 && (Date.now() - startTime) < timeout) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    private handleWorkflowError(event: WorkflowEvent): void {
        console.error(`Workflow error in ${event.name}:`, event.error);

        // Implement error recovery strategies
        this.attemptErrorRecovery(event);
    }

    private handleMemoryError(error: any): void {
        console.error('Memory system error:', error);

        // Implement memory recovery strategies
        this.attemptMemoryRecovery(error);
    }

    private async attemptErrorRecovery(event: WorkflowEvent): Promise<void> {
        // Implement automatic error recovery strategies
        console.log(`Attempting recovery for workflow ${event.name}`);
    }

    private async attemptMemoryRecovery(error: any): Promise<void> {
        // Implement memory system recovery strategies
        console.log('Attempting memory system recovery');
    }
} 