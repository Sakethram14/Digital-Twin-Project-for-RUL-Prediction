def window_to_state(window, sensor_names):
    state_window = []

    for row in window:
        state = {sensor_names[i]: float(row[i]) for i in range(len(sensor_names))}
        state_window.append(state)

    return state_window
