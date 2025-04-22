# Testing Plan for InterruptableFuture

## 1. Inspect files for errors
- [x] Review `interruptable_future.py` for issues
  - Found parameter order mismatch in calls to `_FAILURE_MANAGER.on_event()`
  - In `immediate_interrupt()`, the future is using wrong parameter order
  - In `add_timeout()`, `pg=None` is passed, but the function might require a non-null pg for certain event types
- [x] Review `failure_manager.py` for issues
  - Parameter order in `on_event` is (fut, event_type, id, pg, timeout), but `interruptable_future.py` uses a different order
  - `immediate_interrupt` requires a non-null pg parameter, but doesn't handle the case when it's null properly
  - `rollback` requires a pg, but it's not made optional
- [x] Review `events_manager.py` for issues
  - Abstract `on_event` method signature in `_EventsManager` is incompatible with implementation in `_FailureManager`
  - The signature is `on_event(self, fut: Future[T], event_type: Optional[str]) -> Future[T]`
  - But `_FailureManager` adds parameters `id, pg, timeout`
- [x] Identify any parameter mismatches or logical errors
  - Main issue: Parameter order and signature mismatches between classes
  - `InterruptableFuture.immediate_interrupt()` calls `_FAILURE_MANAGER.on_event()` with (fut, event_type, id, pg, timeout)
  - But `_FailureManager.on_event()` expects (fut, event_type, id, pg=None, timeout=None)
  - This could lead to bugs where pg is accidentally treated as a timeout
  - These issues need to be fixed for proper testing

## 2. Create/modify `interruptable_future_tests_events.py`
- [x] Set up test class structure similar to `interruptable_future_tests_simple.py`
- [x] Import necessary dependencies
- [x] Create setUp method with dummy process group

## 3. Test `immediate_interrupt` behavior
- [x] Test basic interrupt raises ImmediateInterruptException
- [x] Test interrupting an already completed future
- [x] Test that process group's abort is called on interrupt

## 4. Test `add_timeout` behavior
- [x] Test that timeout raises TimeoutError after specified duration
- [x] Test that timeout doesn't occur before specified duration
- [x] Test that future returns normally if completed before timeout
- [x] Test calling add_timeout on already completed future

## 5. Test interaction between methods
- [x] Test immediate_interrupt followed by add_timeout
- [x] Test add_timeout followed by immediate_interrupt
- [x] Test add_timeout called multiple times (should replace timeout)
- [x] Test other potential interactions (in this case, verified all expected cases covered)

## 6. Final verification
- [ ] Run all tests to ensure they pass
- [ ] Review code for clean up or improvements
- [ ] Check that all test cases are covered 