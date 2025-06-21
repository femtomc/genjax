# Claude PR Review Workflows

This directory contains GitHub Actions workflows that integrate Claude for automated PR reviews.

## Available Workflows

### 1. Claude PR Review (`claude-pr-review.yml`)

Automatically reviews PRs when they are opened or updated, or when explicitly triggered.

**Triggers:**
- Automatically on PR open/update
- Manual trigger: Comment `/claude review` on any PR

**Features:**
- Reads CLAUDE.md files for context
- Reviews code changes for correctness, performance, and style
- Checks JAX best practices
- Adds labels based on review findings

### 2. Claude Specialized Review (`claude-specialized-review.yml`)

Provides specialized reviews using different Claude personas from `.claude/commands/`.

**Available Commands:**
- `/claude doctor` - Documentation health check by Dr. Claude
- `/claude renoir` - Code cleanliness review by Renoir
- `/claude shifu` - Git workflow and commit message review by Master Shifu

## Setup Instructions

### 1. Add Anthropic API Key

Add your Anthropic API key as a repository secret:

1. Go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `ANTHROPIC_API_KEY`
4. Value: Your Anthropic API key

### 2. Create Required Labels

The workflows may add these labels to PRs:

```bash
# Create labels using GitHub CLI
gh label create "claude-approved" --description "Approved by Claude review" --color "0e8a16"
gh label create "potential-bug" --description "Claude detected potential issues" --color "d73a4a"
gh label create "jax-control-flow" --description "JAX control flow issues detected" --color "fbca04"
gh label create "performance" --description "Performance considerations" --color "a2eeef"
```

### 3. Ensure Claude Commands Exist

For specialized reviews, ensure these files exist:
- `.claude/commands/doctor.md`
- `.claude/commands/renoir.md`
- `.claude/commands/shifu.md`

## Usage Examples

### Basic Review
```bash
# Automatic on PR creation, or manually trigger:
# Comment on PR: /claude review
```

### Specialized Reviews
```bash
# Check documentation health
# Comment on PR: /claude doctor

# Review code cleanliness
# Comment on PR: /claude renoir

# Review git workflow
# Comment on PR: /claude shifu
```

## Workflow Permissions

The workflows require these permissions:
- `contents: read` - To checkout code
- `pull-requests: write` - To post review comments
- `issues: write` - To add labels

## Customization

### Modify Review Focus

Edit the prompt in `claude-pr-review.yml` to change what Claude focuses on:

```yaml
# Around line 90, modify the review criteria
Please provide a thorough code review focusing on:
1. Your custom criteria here
2. Additional focus areas
```

### Add New Specialized Commands

1. Create a new command file in `.claude/commands/`
2. Add the command to `claude-specialized-review.yml`:

```yaml
# In the if condition
contains(github.event.comment.body, '/claude newcommand') ||

# In the Determine Command step
elif [[ "$comment" == *"/claude newcommand"* ]]; then
  echo "command=newcommand" >> $GITHUB_OUTPUT
  echo "prompt_file=.claude/commands/newcommand.md" >> $GITHUB_OUTPUT
```

## Limitations

- Reviews are limited to 4000 tokens response
- Large diffs may be truncated
- Requires active Anthropic API key with sufficient credits
- Cannot directly approve/reject PRs (only comment)

## Troubleshooting

### Review Not Triggering
- Check workflow runs in Actions tab
- Verify API key is set correctly
- Ensure comment contains exact trigger phrase

### Empty or Error Reviews
- Check API key validity
- Verify Claude command files exist
- Check workflow logs for API errors

### Labels Not Being Added
- Ensure labels exist in repository
- Check bot has permission to add labels
