import { Octokit } from '@octokit/rest';
import type { WorkflowConfig, IssueClassification, ReleasePR, FeedbackItem, LabelingRules, HopeMemorySystem } from '../types.js';

export class GitHubWorkflowManager {
    private octokit: Octokit;
    private config: WorkflowConfig;
    private memory: HopeMemorySystem;
    private labelingRules!: LabelingRules;

    constructor(config: WorkflowConfig, memory: HopeMemorySystem) {
        this.config = config;
        this.memory = memory;
        this.octokit = new Octokit({
            auth: config.integrations.github.authentication.token,
        });

        this.initializeLabelingRules();
    }

    /**
     * Initialize intelligent labeling rules based on memory and configuration
     */
    private initializeLabelingRules(): void {
        this.labelingRules = {
            textPatterns: [
                {
                    pattern: /bug|error|issue|problem|broken|fail/i,
                    labels: ['bug', 'priority: medium'],
                    confidence: 0.8
                },
                {
                    pattern: /feature|enhancement|improvement|add|new/i,
                    labels: ['feature', 'priority: low'],
                    confidence: 0.7
                },
                {
                    pattern: /documentation|docs|readme|guide/i,
                    labels: ['documentation', 'priority: low'],
                    confidence: 0.9
                },
                {
                    pattern: /security|vulnerability|exploit|attack/i,
                    labels: ['security', 'priority: critical'],
                    confidence: 0.95
                },
                {
                    pattern: /performance|slow|optimization|speed/i,
                    labels: ['performance', 'priority: medium'],
                    confidence: 0.8
                }
            ],
            filePatterns: [
                { pattern: '**/*.md', labels: ['documentation'] },
                { pattern: '**/*.test.*', labels: ['testing'] },
                { pattern: '**/package.json', labels: ['dependencies'] },
                { pattern: '**/*.yml', labels: ['ci/cd'] },
                { pattern: '**/src/**', labels: ['component: core'] }
            ],
            userRoles: [
                { role: 'maintainer', defaultLabels: ['maintainer-review'] },
                { role: 'contributor', defaultLabels: ['community'] },
                { role: 'first-time', defaultLabels: ['good first issue'] }
            ],
            contextual: [
                { condition: 'has_reproduction_steps', labels: ['ready-for-triage'] },
                { condition: 'has_test_case', labels: ['well-documented'] },
                { condition: 'breaking_change', labels: ['breaking-change'] }
            ]
        };
    }

    /**
     * Create automatic release PR based on commits and configuration
     */
    async createReleasePR(): Promise<ReleasePR> {
        try {
            // Analyze commits since last release
            const commits = await this.getCommitsSinceLastRelease();
            const versionBump = this.analyzeVersionBump(commits);
            const changelog = await this.generateChangelog(commits);

            // Generate PR content using memory-enhanced intelligence
            const prData = await this.generateReleasePRData(versionBump, changelog, commits);

            // Create the actual PR
            const pr = await this.octokit.pulls.create({
                owner: this.config.repository.owner,
                repo: this.config.repository.name,
                title: prData.title,
                body: prData.body,
                head: `release/${prData.metadata.changeType}-${versionBump}`,
                base: this.config.repository.branch
            });

            // Apply labels and metadata
            await this.octokit.issues.addLabels({
                owner: this.config.repository.owner,
                repo: this.config.repository.name,
                issue_number: pr.data.number,
                labels: prData.labels
            });

            // Store in memory for future reference
            await this.memory.storeWorkflowMemory?.('release_pr', {
                prNumber: pr.data.number,
                versionBump,
                commits: commits.length,
                timestamp: new Date(),
                success: true
            });

            return prData;
        } catch (error) {
            console.error('Error creating release PR:', error);
            throw error;
        }
    }

    /**
     * Process and classify incoming issues
     */
    async processIssue(issueNumber: number): Promise<IssueClassification> {
        try {
            const issue = await this.octokit.issues.get({
                owner: this.config.repository.owner,
                repo: this.config.repository.name,
                issue_number: issueNumber
            });

            // Use memory-enhanced classification
            const classification = await this.classifyIssue(issue.data);

            // Apply intelligent labels
            const labels = await this.generateLabels(issue.data, classification);

            if (labels.length > 0) {
                await this.octokit.issues.addLabels({
                    owner: this.config.repository.owner,
                    repo: this.config.repository.name,
                    issue_number: issueNumber,
                    labels
                });
            }

            // Check for duplicates using memory
            const duplicates = await this.findDuplicateIssues(issue.data);
            if (duplicates.length > 0) {
                await this.handleDuplicateIssue(issueNumber, duplicates);
            }

            // Store classification in memory for learning
            await this.memory.storeWorkflowMemory?.('issue_classification', {
                issueNumber,
                classification,
                labels,
                timestamp: new Date()
            });

            return classification;
        } catch (error) {
            console.error('Error processing issue:', error);
            throw error;
        }
    }

    /**
     * Collect and process feedback from multiple channels
     */
    async collectFeedback(): Promise<FeedbackItem[]> {
        const feedbackItems: FeedbackItem[] = [];

        try {
            // Collect from GitHub issues
            if (this.config.features.feedback.channels.github.issues) {
                const githubFeedback = await this.collectGitHubFeedback();
                feedbackItems.push(...githubFeedback);
            }

            // Collect from discussions
            if (this.config.features.feedback.channels.github.discussions) {
                const discussionFeedback = await this.collectDiscussionFeedback();
                feedbackItems.push(...discussionFeedback);
            }

            // Process and analyze feedback
            for (const item of feedbackItems) {
                item.sentiment = await this.analyzeSentiment(item.content);
                item.topics = await this.extractTopics(item.content);
                item.priority = await this.calculatePriority(item);
                item.actionItems = await this.generateActionItems(item);
            }

            // Store feedback in memory for learning
            await this.memory.storeWorkflowMemory?.('feedback_collection', {
                items: feedbackItems.length,
                timestamp: new Date(),
                sources: feedbackItems.map(f => f.source)
            });

            return feedbackItems;
        } catch (error) {
            console.error('Error collecting feedback:', error);
            throw error;
        }
    }

    /**
     * Run linting and code quality checks
     */
    async runQualityChecks(prNumber?: number): Promise<{
        passed: boolean;
        results: Record<string, any>;
        suggestions: string[];
    }> {
        try {
            const results: Record<string, any> = {};
            const suggestions: string[] = [];
            let passed = true;

            // Syntax checking
            if (this.config.features.linting.levels.syntax.enabled) {
                results.syntax = await this.runSyntaxChecks();
                if (!results.syntax.passed && this.config.features.linting.levels.syntax.failOnError) {
                    passed = false;
                }
            }

            // Style checking
            if (this.config.features.linting.levels.style.enabled) {
                results.style = await this.runStyleChecks();
                suggestions.push(...results.style.suggestions || []);
            }

            // Security scanning
            if (this.config.features.linting.levels.security.enabled) {
                results.security = await this.runSecurityChecks();
                if (results.security.vulnerabilities > 0) {
                    passed = false;
                    suggestions.push('Address security vulnerabilities before merging');
                }
            }

            // Performance analysis
            if (this.config.features.linting.levels.performance.enabled) {
                results.performance = await this.runPerformanceChecks();
                suggestions.push(...results.performance.recommendations || []);
            }

            // Store results in memory for learning
            await this.memory.storeWorkflowMemory('quality_checks', {
                prNumber,
                passed,
                results,
                timestamp: new Date()
            });

            return { passed, results, suggestions };
        } catch (error) {
            console.error('Error running quality checks:', error);
            throw error;
        }
    }

    /**
     * Handle webhook events from GitHub
     */
    async handleWebhook(event: string, payload: any): Promise<void> {
        try {
            switch (event) {
                case 'issues.opened':
                    await this.processIssue(payload.issue.number);
                    break;

                case 'pull_request.opened':
                    await this.runQualityChecks(payload.pull_request.number);
                    break;

                case 'push':
                    if (this.shouldTriggerRelease(payload)) {
                        await this.createReleasePR();
                    }
                    break;

                case 'issue_comment.created':
                    await this.processFeedback(payload.comment);
                    break;

                default:
                    console.log(`Unhandled webhook event: ${event}`);
            }
        } catch (error) {
            console.error(`Error handling webhook ${event}:`, error);
        }
    }

    // Private helper methods

    private async getCommitsSinceLastRelease(): Promise<any[]> {
        // Implementation to get commits since last release
        const releases = await this.octokit.repos.listReleases({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            per_page: 1
        });

        const lastRelease = releases.data[0];
        const since = lastRelease ? lastRelease.created_at : undefined;

        const commits = await this.octokit.repos.listCommits({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            since,
            per_page: 100
        });

        return commits.data;
    }

    private analyzeVersionBump(commits: any[]): 'patch' | 'minor' | 'major' {
        // Analyze commit messages for conventional commits
        const hasBreaking = commits.some(c =>
            c.commit.message.includes('BREAKING CHANGE') ||
            c.commit.message.match(/^[^:]+!:/)
        );

        if (hasBreaking) {return 'major';}

        const hasFeature = commits.some(c =>
            c.commit.message.startsWith('feat:') ||
            c.commit.message.startsWith('feature:')
        );

        if (hasFeature) {return 'minor';}

        return 'patch';
    }

    private async generateChangelog(commits: any[]): Promise<string> {
        const changelog = ['## Changes\n'];

        const features = commits.filter(c => c.commit.message.startsWith('feat:'));
        const fixes = commits.filter(c => c.commit.message.startsWith('fix:'));
        const breaking = commits.filter(c => c.commit.message.includes('BREAKING CHANGE'));

        if (breaking.length > 0) {
            changelog.push('### ⚠️ Breaking Changes\n');
            breaking.forEach(c => changelog.push(`- ${c.commit.message}\n`));
        }

        if (features.length > 0) {
            changelog.push('### ✨ Features\n');
            features.forEach(c => changelog.push(`- ${c.commit.message}\n`));
        }

        if (fixes.length > 0) {
            changelog.push('### 🐛 Bug Fixes\n');
            fixes.forEach(c => changelog.push(`- ${c.commit.message}\n`));
        }

        return changelog.join('');
    }

    private async generateReleasePRData(
        versionBump: string,
        changelog: string,
        commits: any[]
    ): Promise<ReleasePR> {
        const version = await this.calculateNextVersion(versionBump);

        return {
            title: `Release v${version}`,
            body: `# Release v${version}\n\n${changelog}\n\n---\n\nAuto-generated by MCP HOPE Workflow Manager`,
            labels: ['release', `version: ${versionBump}`],
            assignees: [],
            reviewers: [],
            metadata: {
                changeType: versionBump as any,
                affectedComponents: await this.analyzeAffectedComponents(commits),
                testCoverage: await this.calculateTestCoverage(),
                performanceImpact: await this.analyzePerformanceImpact(commits)
            }
        };
    }

    private async classifyIssue(issue: any): Promise<IssueClassification> {
        const content = `${issue.title} ${issue.body}`;

        // Use memory to improve classification over time
        const memoryContext = await this.memory.getRelevantContext('issue_classification', content);

        // Basic classification logic
        const type = this.determineIssueType(content);
        const priority = this.determinePriority(content);
        const complexity = this.determineComplexity(content);
        const component = await this.identifyAffectedComponents(content);

        return {
            type,
            priority,
            complexity,
            component,
            estimatedHours: this.estimateEffort(complexity, type),
            dependencies: await this.findDependencies(content)
        };
    }

    private async generateLabels(issue: any, classification: IssueClassification): Promise<string[]> {
        const labels: string[] = [];

        // Add type label
        labels.push(classification.type);

        // Add priority label
        labels.push(`priority: ${classification.priority}`);

        // Add component labels
        labels.push(...classification.component.map(c => `component: ${c}`));

        // Apply text pattern matching
        const content = `${issue.title} ${issue.body}`;
        for (const rule of this.labelingRules.textPatterns) {
            if (rule.pattern.test(content) && rule.confidence > 0.7) {
                labels.push(...rule.labels);
            }
        }

        return [...new Set(labels)]; // Remove duplicates
    }

    private async findDuplicateIssues(issue: any): Promise<number[]> {
        // Use memory to find similar issues
        const content = `${issue.title} ${issue.body}`;
        const similar = await this.memory.findSimilarContent('issues', content, 0.8);

        return similar.map(s => s.issueNumber).filter(Boolean);
    }

    private async collectGitHubFeedback(): Promise<FeedbackItem[]> {
        const issues = await this.octokit.issues.listForRepo({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            labels: 'feedback',
            state: 'open',
            sort: 'created',
            direction: 'desc',
            per_page: 50
        });

        return issues.data.map(issue => ({
            id: `github-issue-${issue.number}`,
            source: 'github-issues',
            timestamp: new Date(issue.created_at),
            content: `${issue.title}\n${issue.body}`,
            sentiment: 'neutral',
            topics: [],
            priority: 0,
            actionItems: [],
            metadata: {
                issueNumber: issue.number,
                author: issue.user?.login,
                labels: issue.labels.map(l => typeof l === 'string' ? l : l.name)
            }
        }));
    }

    private async collectDiscussionFeedback(): Promise<FeedbackItem[]> {
        // Implementation for GitHub Discussions API
        // This would require GraphQL API calls
        return [];
    }

    private async analyzeSentiment(content: string): Promise<'positive' | 'negative' | 'neutral'> {
        // Simple sentiment analysis - in production, use a proper NLP service
        const positiveWords = ['good', 'great', 'awesome', 'excellent', 'love', 'perfect'];
        const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'broken', 'useless'];

        const words = content.toLowerCase().split(/\s+/);
        const positiveCount = words.filter(w => positiveWords.includes(w)).length;
        const negativeCount = words.filter(w => negativeWords.includes(w)).length;

        if (positiveCount > negativeCount) {return 'positive';}
        if (negativeCount > positiveCount) {return 'negative';}
        return 'neutral';
    }

    private async extractTopics(content: string): Promise<string[]> {
        // Simple topic extraction - in production, use proper NLP
        const topics: string[] = [];
        const topicPatterns = [
            { pattern: /performance|speed|slow|fast/i, topic: 'performance' },
            { pattern: /ui|interface|design|user experience/i, topic: 'ui-ux' },
            { pattern: /bug|error|issue|problem/i, topic: 'bug' },
            { pattern: /feature|functionality|enhancement/i, topic: 'feature' },
            { pattern: /documentation|docs|guide/i, topic: 'documentation' }
        ];

        for (const { pattern, topic } of topicPatterns) {
            if (pattern.test(content)) {
                topics.push(topic);
            }
        }

        return topics;
    }

    private async calculatePriority(item: FeedbackItem): Promise<number> {
        let priority = 0;

        // Sentiment weight
        if (item.sentiment === 'negative') {priority += 3;}
        if (item.sentiment === 'positive') {priority += 1;}

        // Topic weight
        if (item.topics.includes('bug')) {priority += 4;}
        if (item.topics.includes('performance')) {priority += 3;}
        if (item.topics.includes('feature')) {priority += 2;}

        // Recency weight
        const daysOld = (Date.now() - item.timestamp.getTime()) / (1000 * 60 * 60 * 24);
        if (daysOld < 1) {priority += 2;}
        if (daysOld < 7) {priority += 1;}

        return Math.min(priority, 10); // Cap at 10
    }

    private async generateActionItems(item: FeedbackItem): Promise<string[]> {
        const actions: string[] = [];

        if (item.topics.includes('bug')) {
            actions.push('Create bug report issue');
            actions.push('Investigate root cause');
        }

        if (item.topics.includes('feature')) {
            actions.push('Evaluate feature request');
            actions.push('Add to product backlog');
        }

        if (item.sentiment === 'negative' && item.priority > 5) {
            actions.push('Priority response required');
            actions.push('Escalate to team lead');
        }

        return actions;
    }

    // Quality check implementations
    private async runSyntaxChecks(): Promise<any> {
        // Implementation for syntax checking
        return { passed: true, errors: [] };
    }

    private async runStyleChecks(): Promise<any> {
        // Implementation for style checking
        return { passed: true, suggestions: [] };
    }

    private async runSecurityChecks(): Promise<any> {
        // Implementation for security scanning
        return { vulnerabilities: 0, issues: [] };
    }

    private async runPerformanceChecks(): Promise<any> {
        // Implementation for performance analysis
        return { recommendations: [] };
    }

    // Additional helper methods
    private shouldTriggerRelease(payload: any): boolean {
        return payload.ref === `refs/heads/${this.config.repository.branch}` &&
            this.config.features.autoRelease.triggerConditions.commitCount > 0;
    }

    private async processFeedback(comment: any): Promise<void> {
        // Process individual feedback comments
    }

    private determineIssueType(content: string): IssueClassification['type'] {
        if (/bug|error|issue|problem|broken|fail/i.test(content)) {return 'bug';}
        if (/feature|enhancement|improvement|add|new/i.test(content)) {return 'feature';}
        if (/documentation|docs|readme|guide/i.test(content)) {return 'documentation';}
        if (/question|how|help|support/i.test(content)) {return 'question';}
        return 'enhancement';
    }

    private determinePriority(content: string): IssueClassification['priority'] {
        if (/critical|urgent|blocking|broken|security/i.test(content)) {return 'critical';}
        if (/important|high|soon/i.test(content)) {return 'high';}
        if (/low|minor|nice/i.test(content)) {return 'low';}
        return 'medium';
    }

    private determineComplexity(content: string): IssueClassification['complexity'] {
        const complexityIndicators = [
            /refactor|redesign|architecture/i,
            /multiple|several|various/i,
            /integration|api|database/i
        ];

        const complexCount = complexityIndicators.filter(pattern => pattern.test(content)).length;

        if (complexCount >= 2) {return 'complex';}
        if (complexCount === 1) {return 'moderate';}
        if (content.length > 500) {return 'moderate';}
        return 'simple';
    }

    private async identifyAffectedComponents(content: string): Promise<string[]> {
        const components: string[] = [];
        const componentPatterns = [
            { pattern: /api|endpoint|server/i, component: 'api' },
            { pattern: /ui|interface|frontend/i, component: 'ui' },
            { pattern: /database|db|storage/i, component: 'database' },
            { pattern: /auth|login|security/i, component: 'auth' },
            { pattern: /test|testing|spec/i, component: 'testing' }
        ];

        for (const { pattern, component } of componentPatterns) {
            if (pattern.test(content)) {
                components.push(component);
            }
        }

        return components.length > 0 ? components : ['core'];
    }

    private estimateEffort(
        complexity: IssueClassification['complexity'],
        type: IssueClassification['type']
    ): number {
        const baseHours = {
            'trivial': 1,
            'simple': 4,
            'moderate': 16,
            'complex': 40
        };

        const typeMultiplier = {
            'bug': 0.8,
            'feature': 1.2,
            'enhancement': 1.0,
            'question': 0.5,
            'documentation': 0.6
        };

        return Math.round(baseHours[complexity] * typeMultiplier[type]);
    }

    private async findDependencies(content: string): Promise<string[]> {
        // Analyze content for dependency keywords
        const dependencies: string[] = [];

        if (/depends on|requires|blocks|blocked by/i.test(content)) {
            // Extract issue numbers or component names
            const matches = content.match(/#(\d+)/g);
            if (matches) {
                dependencies.push(...matches);
            }
        }

        return dependencies;
    }

    private async calculateNextVersion(bump: string): Promise<string> {
        // Get current version from package.json or latest release
        const releases = await this.octokit.repos.listReleases({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            per_page: 1
        });

        const lastRelease = releases.data[0];
        const currentVersion = lastRelease?.tag_name?.replace('v', '') || '0.0.0';
        const [major, minor, patch] = currentVersion.split('.').map(Number);

        switch (bump) {
            case 'major': return `${major + 1}.0.0`;
            case 'minor': return `${major}.${minor + 1}.0`;
            case 'patch': return `${major}.${minor}.${patch + 1}`;
            default: return `${major}.${minor}.${patch + 1}`;
        }
    }

    private async analyzeAffectedComponents(commits: any[]): Promise<string[]> {
        const components = new Set<string>();

        for (const commit of commits) {
            const files = commit.files || [];
            for (const file of files) {
                if (file.filename.startsWith('src/')) {components.add('core');}
                if (file.filename.includes('test')) {components.add('testing');}
                if (file.filename.includes('docs')) {components.add('documentation');}
                if (file.filename.includes('api')) {components.add('api');}
            }
        }

        return Array.from(components);
    }

    private async calculateTestCoverage(): Promise<number> {
        // Implementation would integrate with coverage tools
        return 85; // Placeholder
    }

    private async analyzePerformanceImpact(commits: any[]): Promise<string> {
        // Analyze commits for performance-related changes
        const perfKeywords = ['performance', 'optimization', 'cache', 'memory', 'speed'];
        const hasPerfChanges = commits.some(c =>
            perfKeywords.some(keyword => c.commit.message.toLowerCase().includes(keyword))
        );

        return hasPerfChanges ? 'Performance improvements included' : 'No significant performance impact';
    }

    private async handleDuplicateIssue(issueNumber: number, duplicates: number[]): Promise<void> {
        const comment = `This issue appears to be a duplicate of ${duplicates.map(d => `#${d}`).join(', ')}. 
    
Please check the existing issues before creating new ones. If this is not a duplicate, please provide additional context to help us understand the difference.`;

        await this.octokit.issues.createComment({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            issue_number: issueNumber,
            body: comment
        });

        await this.octokit.issues.addLabels({
            owner: this.config.repository.owner,
            repo: this.config.repository.name,
            issue_number: issueNumber,
            labels: ['duplicate']
        });
    }
} 