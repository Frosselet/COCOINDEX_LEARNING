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
  - COLPALI-800 (Governance & Lineage) ✅
  - COLPALI-900 (Lambda Deployment) ✅
- **Active Story**: Ready for COLPALI-1000 (Testing & Validation)
- **Current Branch**: `feature/COLPALI-900-lambda-deployment`
- **Next**: Comprehensive testing with 15 PDF samples and performance benchmarking

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

## Testing Framework & Guidelines

### Philosophy: 100% Pass Rate Required

**Tests exist to validate implementation correctness. Anything less than 100% pass rate is unacceptable for merge.**

- A 70% pass rate means 30% of functionality is broken
- Modifying tests to "accept failure" is forbidden - this hides bugs
- Tests should be precise, not lenient
- When tests fail, fix the CODE, not the tests (unless tests are genuinely wrong)

### Test Organization by JIRA Structure

Tests follow the JIRA story/task hierarchy:

```
tests/
├── unit/                           # Unit tests for individual tasks
│   ├── COLPALI-901/               # Task-level tests
│   │   ├── test_model_optimizer.py
│   │   └── test_quantization.py
│   ├── COLPALI-902/
│   │   └── test_resource_manager.py
│   └── ...
├── integration/                    # Integration tests for stories
│   ├── test_colpali_900_integration.py  # Story-level integration
│   └── ...
└── e2e/                           # End-to-end tests
    └── test_full_pipeline.py
```

### Test Planning Template

**Before writing ANY code for a task, create the test plan:**

```markdown
## Test Plan: COLPALI-XXX - [Task Name]

### Unit Tests Required
| Test Case | Description | Expected Behavior | Priority |
|-----------|-------------|-------------------|----------|
| test_xxx_happy_path | Normal operation | Returns expected result | P0 |
| test_xxx_edge_case_empty | Empty input | Handles gracefully | P1 |
| test_xxx_error_condition | Invalid input | Raises appropriate error | P1 |

### Dependencies & Environment Requirements
- Required packages: [list]
- Environment variables: [list]
- External services: [list]

### Integration Test Scope (Story Level)
- Component A + Component B interaction
- End-to-end flow validation
```

### Test Execution Protocol

**MANDATORY: Execute tests at EVERY step, not just before merge.**

1. **After writing each test file:**
   ```bash
   PYTHONPATH=. pytest tests/path/to/new_test.py -v
   ```

2. **After implementing each task:**
   ```bash
   PYTHONPATH=. pytest tests/unit/COLPALI-XXX/ -v
   ```

3. **After completing a story:**
   ```bash
   PYTHONPATH=. pytest tests/integration/test_colpali_XXX_integration.py -v
   ```

4. **Before merge (full suite):**
   ```bash
   PYTHONPATH=. pytest tests/ -v --tb=short
   ```

### Test Failure Analysis Template

**When tests fail, document using this exact template:**

```markdown
## Test Failure Report

**Date:** [YYYY-MM-DD]
**Task:** COLPALI-XXX
**Test File:** tests/path/to/test.py

### Failure Summary
| Metric | Value |
|--------|-------|
| Total Tests | XX |
| Passed | XX |
| Failed | XX |
| Pass Rate | XX% |

### Failed Tests Detail

#### 1. test_name_here
- **Error Message:** [exact error]
- **Root Cause:** [code bug / environment issue / test design flaw / missing dependency]
- **Criticality:** [BLOCKER / HIGH / MEDIUM / LOW]
- **Impact:** [What functionality is affected]
- **Resolution:** [Fix applied OR why it's acceptable to defer]

#### 2. test_name_here
[repeat for each failure]

### Criticality Definitions
- **BLOCKER:** Core functionality broken. MUST fix before proceeding.
- **HIGH:** Important feature affected. Should fix before merge.
- **MEDIUM:** Edge case or minor feature. Document and track.
- **LOW:** Non-critical, environment-specific. Document for future.

### Decision
- [ ] All failures resolved - proceed with merge
- [ ] Deferred failures documented with tracking issue
- [ ] BLOCKED - must fix before continuing
```

### Forbidden Practices

1. **NO test modification to hide failures:**
   ```python
   # FORBIDDEN: Making test accept failure
   if not result:
       pass  # This hides the bug!

   # CORRECT: Test should fail if expectation not met
   assert result, "Expected result to be truthy"
   ```

2. **NO skipping tests without documentation:**
   ```python
   # FORBIDDEN
   @pytest.mark.skip
   def test_something():
       ...

   # ALLOWED (with clear reason)
   @pytest.mark.skip(reason="Requires FBGEMM engine not available in CI - tracked in COLPALI-999")
   def test_quantization_specific():
       ...
   ```

3. **NO "lenient" assertions:**
   ```python
   # FORBIDDEN: Accepting any outcome
   if quantization_worked:
       assert size_reduced
   else:
       pass  # Silently accepts failure

   # CORRECT: Be explicit about environment requirements
   @pytest.mark.skipif(not QUANTIZATION_AVAILABLE, reason="Requires FBGEMM")
   def test_quantization_reduces_size():
       assert quantized_size < original_size
   ```

### Environment-Specific Test Handling

When tests depend on specific environments (GPU, specific libraries, etc.):

1. **Use pytest markers:**
   ```python
   import pytest

   FBGEMM_AVAILABLE = check_fbgemm()

   @pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="Requires FBGEMM quantization engine")
   def test_int8_quantization():
       # Test only runs when FBGEMM is available
       ...
   ```

2. **Document in test file header:**
   ```python
   """
   Tests for model quantization.

   Environment Requirements:
   - PyTorch with FBGEMM backend (Intel CPUs or specific builds)
   - 8GB+ RAM for large model tests

   Skip Conditions:
   - Tests marked with @skipif will skip gracefully if FBGEMM unavailable
   """
   ```

3. **CI/CD configuration should match production environment**

### Pre-Merge Checklist

- [ ] All tests executed with `pytest tests/ -v`
- [ ] 100% pass rate achieved (or failures documented per template)
- [ ] No test modifications made to hide failures
- [ ] Environment-specific skips properly documented with markers
- [ ] Test failure report generated if any failures occurred
- [ ] BLOCKER/HIGH failures resolved before merge
- [ ] MEDIUM/LOW failures tracked with issues

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
