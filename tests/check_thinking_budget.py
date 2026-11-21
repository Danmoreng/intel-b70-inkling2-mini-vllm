"""Check Inkling's per-request thinking budget across message and DSpark boundaries."""
import sys
import torch

START, END, TEXT, EOS = 1, 2, 3, 4


def check(lib):
    torch.ops.load_library(lib)
    device = torch.device("xpu")
    state = torch.full((2, 3), -1, dtype=torch.int32, device=device)
    history = torch.zeros(2, 16384, dtype=torch.int32, device=device)

    def run(slot, prompt, generated, draft, budget, expected):
        tokens = prompt + generated
        history[slot, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        length = torch.zeros(2, dtype=torch.int32, device=device)
        length[slot] = len(tokens)
        ids = torch.tensor([tokens[-1], *draft], dtype=torch.int32, device=device)
        scores = torch.zeros(len(ids), 8, device=device)
        torch.ops.inkling2_xpu.thinking_budget(scores, history, length, ids, state, slot, len(prompt),
                                              budget, START, END, EOS)
        result = [("close" if r[END] > 1e8 else "block" if r[START].isneginf() else "open")
                  for r in scores.cpu()]
        assert result == expected, (result, expected)

    # Closed reasoning in prior turns doesn't consume the new allowance.
    prior = [START, TEXT, TEXT, END, TEXT]
    committed = [START, TEXT, END, TEXT, START, TEXT]
    run(1, prior, committed, [TEXT, END, START], 3, ["open", "close", "block", "close"])
    # The proposed third token was rejected. Commit a different continuation.
    run(1, prior, committed + [END], [START, TEXT], 3, ["open", "open", "close"])
    # Preemption/rollback resets an advanced cursor to the original prompt.
    run(1, prior, [START], [TEXT, TEXT, TEXT], 3, ["open", "open", "open", "close"])
    state[0].fill_(-1)
    run(0, prior, [START, TEXT, EOS], [TEXT, START, TEXT], 2, ["open", "open", "open", "close"])
    state[0].fill_(-1)
    # An unfinished span at the prompt tail continues into the new output.
    run(0, [TEXT, START, TEXT, TEXT], [], [TEXT], 3, ["open", "close"])
    # A slot reused by a new request gets a fresh counter.
    state[0].fill_(-1)
    run(0, prior, [], [START], 0, ["block", "close"])
    state[0].fill_(-1)
    run(0, prior, [START]+[TEXT]*4096+[END,START]+[TEXT]*4095, [TEXT,END], 8192,
        ["open", "close", "block"])
    print("Passed cumulative 8192, zero budget, proposal rollback, slot reuse and prompt-tail checks.")


if __name__ == "__main__":
    check(sys.argv[1])
