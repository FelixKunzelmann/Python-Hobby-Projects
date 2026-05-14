def search_history(history: str, input_str: str, max_offset: int) -> str:
    compressed = []
    input_index = 0

    while input_index < len(input_str):
        search_start = max(0, len(history) - max_offset)
        search_history = history[search_start:]

        max_length = 0
        match_offset = 0

        for length in range(len(input_str) - input_index, 0, -1):
            candidate = input_str[input_index:input_index + length]
            if candidate in search_history:
                pos = search_history.rfind(candidate)

                match_offset = len(search_history) - pos
                max_length = length
                break

        if max_length > 0:

            compressed.append(str(match_offset))
            compressed.append(str(max_length))
            input_index += max_length

        else:

            compressed.append('0')
            compressed.append(input_str[input_index])
            input_index += 1

    return "".join(compressed)


print(search_history("ABCDABC", "ABCDXX", 5))
