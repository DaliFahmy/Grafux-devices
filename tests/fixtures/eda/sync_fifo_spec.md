# sync_fifo — specification

An 8-entry, 8-bit synchronous FIFO with a single clock.

## Interface

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk` | in | 1 | Rising-edge clock. |
| `rst_n` | in | 1 | Active-low asynchronous reset. |
| `wr_en` | in | 1 | Request to push `data_in` on this edge. |
| `rd_en` | in | 1 | Request to pop on this edge. |
| `data_in` | in | 8 | Data to push. |
| `data_out` | out | 8 | Data popped on the previous accepted read. |
| `full` | out | 1 | The FIFO holds DEPTH entries. |
| `empty` | out | 1 | The FIFO holds no entries. |
| `count` | out | 4 | Number of entries currently held. |

## Behaviour

1. **Reset.** While `rst_n` is low, `count` is 0, `empty` is 1, `full` is 0 and
   `data_out` is 0.
2. **Push.** A write is accepted on a rising edge when `wr_en` is high and `full`
   is low; `count` increments.
3. **Pop.** A read is accepted on a rising edge when `rd_en` is high and `empty`
   is low; `data_out` presents the oldest unread entry and `count` decrements.
4. **Order.** Entries are returned in the order they were pushed (FIFO).
5. **Full.** `full` is high exactly when `count == 8`, and never before.
6. **Empty.** `empty` is high exactly when `count == 0`.
7. **Simultaneous push and pop** on the same edge leave `count` unchanged and are
   both accepted, provided the FIFO is neither full (for the push) nor empty (for
   the pop).

## Must NOT happen

- The FIFO must never accept a write while `full` is high; `count` must never
  exceed 8.
- The FIFO must never accept a read while `empty` is high; `count` must never
  wrap below 0.
