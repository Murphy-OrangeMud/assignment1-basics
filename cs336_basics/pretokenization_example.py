from collections import defaultdict
import os
from typing import BinaryIO, List, Dict, Tuple
import multiprocessing
import regex as re

num_processes = 128

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token_regex: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token_regex, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            chunks = re.split(split_special_token_regex, mini_chunk)
            found_at = len(chunks[0])
            if found_at != mini_chunk:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def run_pretokenization(input_path: str, special_tokens: List[str]) -> Dict[bytes, int]:
    ## Usage
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, r"|".join(special_tokens).encode("utf-8"))

        manager = multiprocessing.Manager()

        dictionary = manager.dict()

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        def func(start: int, end: int):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            match_iter = re.finditer(PAT, chunk)

            part_dict = defaultdict(int)

            for match in match_iter:
                part_dict[match.group().lower().strip()] += 1

            for key in part_dict.keys():
                try:
                    dictionary[tuple(letter.encode('utf-8') for letter in key)] += part_dict[key]
                except KeyError:
                    dictionary[tuple(letter.encode('utf-8') for letter in key)] = 0
                    dictionary[tuple(letter.encode('utf-8') for letter in key)] += part_dict[key]

        procs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            proc = multiprocessing.Process(target=func, args=(start, end))
            procs.append(proc)
            proc.start()
            break

        for proc in procs:
            proc.join()

    return dictionary


def test_pretokenization():
    input_path = "/home/murphy/Repos/data/TinyStories-valid.txt"
    print(run_pretokenization(input_path, ["<|endoftext|>"]))


if __name__ == "__main__":
    test_pretokenization()
