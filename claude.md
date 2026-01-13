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
   - Push the updated `main` branch to remote: `git push origin main`
   - Delete the feature branch locally and remotely (if pushed)
   - Immediately create new branch for the next JIRA story
   - Continue this cycle through all 11 stories in the implementation plan

### Current Status Tracking

- **Completed Stories**:
  - COLPALI-100 (Core Infrastructure & Docker Foundation) ✅
  - COLPALI-200 (Document Processing Pipeline) ✅
  - COLPALI-300 (ColPali Vision Integration) ✅
  - COLPALI-400 (Qdrant Vector Storage Integration) ✅
  - COLPALI-500 (BAML Schema System) ✅
  - COLPALI-600 (Extraction & Validation) ✅
  - COLPALI-700 (Output Management) ✅
- **Active Story**: COLPALI-800 (Governance & Lineage)
- **Current Branch**: `main` (ready for next feature branch)
- **Next**: Implement transformation lineage tracking and governance validation rules

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

### Git Workflow Validation

**CRITICAL**: Before starting any new story, validate the git workflow:

1. **Pre-Development Checklist**:
   - ✅ Confirm you are on `main` branch
   - ✅ Create feature branch: `feature/COLPALI-XXX-short-description`
   - ✅ NEVER commit directly to `main` branch
   - ✅ Reference JIRA ticket number in all commits

2. **Pre-Merge Validation**:
   - ✅ All commits are on feature branch, not main
   - ✅ Git log shows proper branch/merge structure
   - ✅ All acceptance criteria completed on feature branch
   - ✅ Create proper merge commit with detailed message

3. **Post-Merge Cleanup**:
   - ✅ Push updated main branch to origin: `git push origin main`
   - ✅ Delete feature branch locally: `git branch -d feature/COLPALI-XXX-xxx`
   - ✅ Delete feature branch remotely (if pushed): `git push origin --delete feature/COLPALI-XXX-xxx`
   - ✅ Verify git graph shows clean branch/merge pattern

### Development Integrity Requirements

**CRITICAL**: No shortcuts or simplifications when facing technical challenges.

- **Full Implementation**: Every feature must be implemented exactly as specified in the JIRA plan
- **Systematic Problem Solving**: Address dependency issues, import problems, and technical challenges head-on
- **Plan Updates**: If changes are truly necessary, explicitly update the JIRA plan and ADR with clear reasoning
- **Test Organization**: All tests must be properly organized in structured folders with clear naming
- **Consistency**: Maintain full consistency between documentation, implementation, and testing
- **No Workarounds**: Avoid temporary solutions or simplified implementations that deviate from the plan

*When encountering technical difficulties, the solution is better implementation, not easier requirements.*

---

*This document serves as our development contract and should be referenced throughout the project lifecycle.*
