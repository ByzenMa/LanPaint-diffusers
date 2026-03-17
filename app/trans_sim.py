def simulate_channel_process(r_channel_start, r_channel_end, g_channel_start, g_channel_end, b_channel_start, b_channel_end):
    # Ensure the start and end are within [0,1] and sorted
    points = sorted([0, 1, r_channel_start, r_channel_end, g_channel_start, g_channel_end, b_channel_start, b_channel_end])
    
    # Remove duplicates while preserving order
    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
            
    channel_list = []
    channel_start_end_list = [[], []]
    
    # Initialize previous point as the first point in the timeline (always 0)
    prev_point = unique_points[0]
    
    for i in range(1, len(unique_points)):
        current_point = unique_points[i]
        
        # Determine which channels are active during this interval
        active_channels = ''
        if r_channel_start <= prev_point < r_channel_end: active_channels += 'r'
        if g_channel_start <= prev_point < g_channel_end: active_channels += 'g'
        if b_channel_start <= prev_point < b_channel_end: active_channels += 'b'
        
        # If no channels are active, use 'b' to indicate a blank channel
        if not active_channels:
            active_channels = 'b'
        
        # Add the active channels and their corresponding time intervals to the lists
        channel_list.append(active_channels)
        channel_start_end_list[0].append(prev_point)
        channel_start_end_list[1].append(current_point)
        
        # Update the previous point to the current one
        prev_point = current_point
    
    return channel_list, channel_start_end_list

# Given input parameters
r_channel_start, r_channel_end = 0.1, 0.9
g_channel_start, g_channel_end = 0, 0.3
b_channel_start, b_channel_end = 0.3, 0.6

# Call the simulation function with the given parameters and print the output
channel_list, channel_start_end_list = simulate_channel_process(
    r_channel_start, r_channel_end,
    g_channel_start, g_channel_end,
    b_channel_start, b_channel_end
)

print("Input Parameters:")
print(f"r_channel_start={r_channel_start}, r_channel_end={r_channel_end}")
print(f"g_channel_start={g_channel_start}, g_channel_end={g_channel_end}")
print(f"b_channel_start={b_channel_start}, b_channel_end={b_channel_end}")

# Output the results
print("Output Results:")
print("channel_list:", channel_list)
print("channel_start_end_list:", channel_start_end_list)