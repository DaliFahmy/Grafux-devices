"""A hand-written cocotb testbench for sync_fifo.

Deliberately NOT generated: it exists so the cocotb runner, the results parser and
the verify image can be tested without an LLM in the loop, and so there is a known
-good testbench to compare a generated one against. It must PASS
sync_fifo_good.v and FAIL sync_fifo_bug_full_flag.v.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

DEPTH = 8


async def reset_dut(dut):
    """Start the clock and apply reset; leaves the DUT idle and empty."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.rd_en.value = 0
    dut.data_in.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push(dut, value):
    dut.wr_en.value = 1
    dut.data_in.value = value
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0


async def pop(dut):
    dut.rd_en.value = 1
    await RisingEdge(dut.clk)
    dut.rd_en.value = 0
    await RisingEdge(dut.clk)
    return int(dut.data_out.value)


@cocotb.test()
async def test_reset_values(dut):
    """Spec 1: reset leaves the FIFO empty, not full, count 0."""
    await reset_dut(dut)
    assert int(dut.count.value) == 0, f"count must be 0 after reset, got {int(dut.count.value)}"
    assert int(dut.empty.value) == 1, "empty must assert after reset (spec: reset)"
    assert int(dut.full.value) == 0, "full must not assert after reset (spec: reset)"


@cocotb.test()
async def test_push_pop_roundtrip(dut):
    """Spec 4: entries come back in the order they went in."""
    await reset_dut(dut)
    values = [0x11, 0x22, 0x33]
    for value in values:
        await push(dut, value)
    for expected in values:
        got = await pop(dut)
        assert got == expected, f"FIFO order violated: expected {expected:#x}, got {got:#x}"


@cocotb.test()
async def test_full_asserts_at_depth(dut):
    """Spec 5: full is high exactly at count == DEPTH — the seeded bug's test."""
    await reset_dut(dut)
    for i in range(DEPTH):
        await push(dut, i)
    await RisingEdge(dut.clk)
    assert int(dut.count.value) == DEPTH, (
        f"count must be {DEPTH} after {DEPTH} writes, got {int(dut.count.value)}"
    )
    assert int(dut.full.value) == 1, (
        f"full must assert after {DEPTH} writes (spec 5: full iff count == {DEPTH}), "
        f"got {int(dut.full.value)} with count={int(dut.count.value)}"
    )


@cocotb.test()
async def test_empty_tracks_count(dut):
    """Spec 6: empty is high exactly when count is 0."""
    await reset_dut(dut)
    await push(dut, 0xAB)
    await RisingEdge(dut.clk)
    assert int(dut.empty.value) == 0, "empty must deassert once an entry is held (spec 6)"
    await pop(dut)
    assert int(dut.empty.value) == 1, "empty must reassert once the FIFO drains (spec 6)"


@cocotb.test()
async def test_count_must_not_exceed_depth(dut):
    """Must-NOT: overfilling can never push count past DEPTH."""
    await reset_dut(dut)
    for i in range(DEPTH + 4):
        await push(dut, i & 0xFF)
        count = int(dut.count.value)
        assert count <= DEPTH, (
            f"count reached {count} on write {i + 1}: the FIFO accepted a write while "
            f"full (must-NOT: count never exceeds {DEPTH})"
        )


@cocotb.test()
async def test_random_traffic_matches_reference(dut):
    """Spec 2/3/4 under random traffic, checked against a Python model."""
    random.seed(1234)
    await reset_dut(dut)
    model = []
    for _ in range(60):
        if len(model) < DEPTH and random.random() < 0.6:
            value = random.randrange(256)
            await push(dut, value)
            model.append(value)
        elif model:
            expected = model.pop(0)
            got = await pop(dut)
            assert got == expected, (
                f"random traffic: expected {expected:#x}, got {got:#x} "
                f"(spec 4: FIFO order)"
            )
        assert int(dut.count.value) == len(model), (
            f"count is {int(dut.count.value)} but {len(model)} entries are outstanding"
        )
