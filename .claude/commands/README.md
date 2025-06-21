# Claude Commands for GenJAX

This directory contains specialized command personas for different aspects of GenJAX development. Each command activates a specific expert mode optimized for particular tasks.

## Available Commands

### 🧘 `/chikhai-bardo` - McCoy, The Probabilistic Sage
**Purpose**: Deep codebase orientation and philosophical guidance
**Use when**: Starting work on GenJAX, needing to understand core concepts, or seeking wisdom about JAX patterns
**Style**: Zen koans and archaic vernacular (patient but cryptic)
**Version**: v2 (optimized for Claude 4)

### 🔬 `/george` - Model-Inference Co-Design Expert
**Purpose**: Systematic development of probabilistic models with inference considerations
**Use when**: Creating new examples, optimizing model-inference pairs, or debugging convergence issues
**Style**: Direct, precise, metrics-driven
**Version**: v2 (optimized for Claude 4)

### 📝 `/improver` - Prompt Enhancement Specialist
**Purpose**: Transform basic prompts into comprehensive, structured instructions
**Use when**: Creating new commands, improving documentation, or optimizing Claude interactions
**Style**: Systematic, template-driven, meta-analytical
**Version**: v2 (optimized for Claude 4)

## Command Design Principles

All commands follow Claude 4 best practices:

1. **XML Structure**: Heavy use of XML tags for clarity
2. **Explicit Workflows**: Step-by-step processes with checkpoints
3. **Concrete Examples**: Real code/interaction examples
4. **Clear Constraints**: Explicit dos and don'ts
5. **Parallel Processing**: Instructions for concurrent operations
6. **Decision Trees**: Conditional logic for different scenarios

## Creating New Commands

Use `/improver` to enhance your initial command idea, ensuring:
- Clear role definition with specific expertise
- Structured workflow with phases
- Explicit constraints and guidelines
- Response examples
- Edge case handling

## Version History

- **v2 Series**: Optimized for Claude 4 (June 2025)
  - Added XML structuring
  - Enhanced parallel task instructions
  - Explicit constraint sections
  - Better example integration

- **v1 Series**: Original commands (deprecated)
  - Basic role definitions
  - Linear instructions
  - Limited structure

## Usage Tips

1. Commands can be combined - start with `/chikhai-bardo` for orientation, then switch to `/george` for implementation
2. Each command maintains its persona throughout the conversation
3. Commands include specific references to GenJAX's CLAUDE.md documentation
4. All commands emphasize GenJAX-specific best practices (JAX control flow, pixi commands, etc.)

## Maintenance

When updating commands:
1. Test with real GenJAX tasks
2. Ensure compatibility with latest CLAUDE.md guidelines
3. Update version number and this README
4. Consider backward compatibility for ongoing conversations
