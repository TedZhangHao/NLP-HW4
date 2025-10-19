#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
from ast import If
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Literal

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def best_parse(self) -> str: # added
        """Return the best parse as an S-expression, or a message if no parse was found."""
        # Find all complete items in the last column that span the whole input
        # and have the start symbol on their left-hand side.
        candidates = [item for item in self.cols[-1].all()
                      if item.rule.lhs == self.grammar.start_symbol
                      and item.next_symbol() is None
                      and item.start_position == 0]
        if not candidates:
            return f"# No parse: {' '.join(self.tokens)}"
        # Pick the candidate with the lowest weight
        best_item = min(candidates, key=lambda item: self.cols[-1]._weight[item])
        log.debug(f"Best parse has weight {self.cols[-1]._weight[best_item]}")
        # Reconstruct the parse tree from backpointers
        return self.build_trees(best_item, len(self.tokens))

    def build_trees(self, item: Item, end: int) -> str: # added
        """Build an S-expression for a complete `item` that ends at column `end`.
        Walk back through backpointers, collecting children left-to-right.
        """
        children_rev: List[str] = []
        cur_item, cur_end = item, end
        while True:
            bp = self.cols[cur_end]._backptr[cur_item] # type: ignore
            if bp.kind == 'PREDICT':
                break  # reached the start of this rule's derivation (dot at 0)
            elif bp.kind == 'SCAN':
                # Consumed one terminal; add it and step back one column
                children_rev.append(bp.terminal) # type: ignore
                cur_item = bp.parent_item
                cur_end -= 1
            elif bp.kind == 'ATTACH':
                # Attached a completed child; build its subtree and step back to parent prefix
                children_rev.append(self.build_trees(bp.child_item, bp.child_end)) # type: ignore
                # Parent prefix ended where the child began
                cur_item = bp.parent_item
                cur_end = bp.child_item.start_position # type: ignore
            else:
                raise ValueError(f"Unknown backpointer kind: {bp.kind}")

        children = list(reversed(children_rev))
        return f"({item.rule.lhs} {' '.join(children)})" if children else f"({item.rule.lhs})"


    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)                      

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            new_weight = rule.weight
            new_backptr = Backptr(kind='PREDICT', parent_item=None, parent_end=None,
                                  child_item=None, child_end=None, terminal=None)
            self.cols[position].push_or_move(new_item, new_weight, new_backptr)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            new_weight = self.cols[position]._weight[item]  # no additional weight for scanning
            new_backptr = Backptr(kind='SCAN', parent_item=item, parent_end=position,
                                  child_item=None, child_end=None, terminal=self.tokens[position])
            self.cols[position + 1].push_or_move(new_item, new_weight, new_backptr)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        # Use the column's index of waiting customers to avoid linear scan
        child = item
        child_weight = self.cols[position]._weight[child]
        child_end = position
        for customer in self.cols[mid]._waiting.get(item.rule.lhs, ()):
            new_item = customer.with_dot_advanced()
            new_weight = self.cols[mid]._weight[customer] + child_weight
            # Create backpointer from new_item to customer and child
            self.cols[position].push_or_move(new_item, new_weight, Backptr(kind='ATTACH', parent_item=customer, parent_end=position,
                                                                             child_item=child, child_end=child_end, terminal=None))
            log.debug(f"\tAttached to get: {new_item} in column {position}")
            self.profile["ATTACH"] += 1


class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    >>> a = Agenda()
    >>> a.push(3)
    >>> a.push(5)
    >>> a.push(3)   # duplicate ignored
    >>> a
    Agenda([]; [3, 5])
    >>> a.pop()
    3
    >>> a
    Agenda([3]; [5])
    >>> a.push(3)   # duplicate ignored
    >>> a.push(7)
    >>> a
    Agenda([3]; [5, 7])
    >>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._next = 0                     # index of first item that has not yet been popped
        # Index of items by their next symbol (terminal or nonterminal), for fast lookup
        # of customers during attach. Items with next_symbol() is None are not indexed.
        self._waiting: Dict[str, List[Item]] = {}

        self._weight: Dict[Item, float] = {} # stores weight of each item
        self._backptr: Dict[Item, Backptr] = {} # stores backpointers for each item

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    # def push(self, item: Item) -> None:
    #     """Add (enqueue) the item, unless it was previously added."""
    #     if item not in self._index:    # O(1) lookup in hash table
    #         self._items.append(item)
    #         self._index[item] = len(self._items) - 1
    #         sym = item.next_symbol()
    #         if sym is not None:
    #             self._waiting.setdefault(sym, []).append(item)

    def push_or_move(self, item: Item, weight: float, backptr: Backptr) -> None:
        """Add (enqueue) the item, unless it was previously added.
        If it was previously added with a higher weight, update its weight and backpointer."""
        idx = self._index.get(item)
        if idx is None:
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            sym = item.next_symbol()
            if sym is not None:
                self._waiting.setdefault(sym, []).append(item)
            self._weight[item] = weight
            self._backptr[item] = backptr
        elif weight < self._weight[item]:
            self._weight[item] = weight
            self._backptr[item] = backptr
            if idx < self._next:
                # if item already popped (idx < _next), swap it with _items[_next-1],
                # update the two _index entries, then do _next -= 1 to “unpop” it in O(1).
                last_idx = self._next - 1
                if idx != last_idx:  # no need to swap if it's already at _next-1
                    self._items[idx], self._items[last_idx] = self._items[last_idx], self._items[idx]
                    self._index[self._items[idx]] = idx
                    self._index[self._items[last_idx]] = last_idx
                self._next -= 1

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us declare that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)} ({self.weight})"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides

@dataclass
class Backptr:
    """Backpointer information for an item."""
    kind: Literal['PREDICT', 'SCAN', 'ATTACH']
    parent_item: Optional[Item]
    parent_end: Optional[int]
    child_item: Optional[Item]
    child_end: Optional[int]
    terminal: Optional[str]

    def __post_init__(self):
        # Enforce field requirements based on kind
        if self.kind == 'SCAN':
            assert self.parent_item is not None
            assert self.parent_end is not None
            assert self.terminal is not None
        elif self.kind == 'ATTACH':
            assert self.parent_item is not None
            assert self.parent_end is not None
            assert self.child_item is not None
            assert self.child_end is not None
        elif self.kind == 'PREDICT':
            pass
        else:
            raise ValueError(f"Unknown backpointer kind: {self.kind}")

    def __repr__(self) -> str:
        if self.kind == 'PREDICT':
            return f"PREDICT"
        elif self.kind == 'SCAN':
            return f"SCAN '{self.terminal}'"
        elif self.kind == 'ATTACH':
            return f"ATTACH from {self.child_item} at {self.child_end} to {self.parent_item} at {self.parent_end}"
        else:
            return "UNKNOWN BACKPTR"

def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                print(chart.best_parse())
                
                # print the result
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    # import doctest
    # doctest.testmod(verbose=False)   # run tests
    main()
