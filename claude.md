# Claude Code Development Rules

## Project Development Workflow

This document establishes the development workflow and rules for the ColPali-BAML Vision Processing Engine project.

### Core Development Rules

1. **JIRA Plan Adherence**
   - We strictly follow the JIRA implementation plan (`docs/jira-implementation-plan.md`)
   - The plan must be kept up-to-date throughout development
   - Any changes to scope, timeline, or approach must be reflected in the plan
   - Reference specific JIRA ticket numbers in all commits and branches

2. **Git Branch Strategy**
   - Each JIRA story gets its own dedicated git branch
   - Branch naming convention: `feature/COLPALI-XXX-short-description`
   - Example: `feature/COLPALI-100-core-infrastructure`
   - No direct commits to `main` branch

3. **Story Completion Criteria**
   - Continue developing until **ALL** success criteria are validated
   - Each story's acceptance criteria must be fully met before merging
   - Use TodoWrite tool to track progress on success criteria
   - Test all functionality thoroughly before considering story complete

4. **Pre-Merge Requirements**
   - Before merging any story branch:
     - Update the JIRA implementation plan's "Definition of Done" section for the completed story
     - Mark all tasks as completed with ✅ checkmarks
     - Update the README file with:
       - Detailed explanation of what was implemented
       - How to use the new functionality
       - Any configuration changes or setup requirements
       - Updated installation/deployment instructions
     - Ensure all tests pass
     - Verify Docker builds work correctly
     - Validate all story acceptance criteria

5. **Merge and Iteration Process**
   - Once story is complete and README updated, merge to `main`
   - Immediately create new branch for the next JIRA story
   - Continue this cycle through all 11 stories in the implementation plan

### Current Status Tracking

- **Active Story**: COLPALI-100 (Core Infrastructure & Docker Foundation)
- **Current Branch**: `main` (initial setup)
- **Next Branch**: `feature/COLPALI-100-core-infrastructure`

### Story Progress Tracking

Use the TodoWrite tool to maintain visibility into:
- Story acceptance criteria completion
- Technical implementation tasks
- Testing and validation status
- Documentation requirements

### Quality Standards

- All Docker builds must succeed (dev, lambda, jupyter)
- All tests must pass
- Code must follow established architectural patterns
- Documentation must be complete and accurate
- Memory and performance requirements must be met

---

*This document serves as our development contract and should be referenced throughout the project lifecycle.*