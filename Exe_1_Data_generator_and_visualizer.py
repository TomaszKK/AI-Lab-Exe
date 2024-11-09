import numpy as np
import pandas as pd
import streamlit as st

def generate_class_data(num_modes, num_samples):
    x_data, y_data = [], []

    for _ in range(num_modes):
        mean_x = np.random.uniform(-1, 1)
        mean_y = np.random.uniform(-1, 1)
        std_var_x = np.random.uniform(0.05, 0.1)
        std_var_y = np.random.uniform(0.05, 0.1)
        x_data.extend(np.random.normal(mean_x, std_var_x, num_samples))
        y_data.extend(np.random.normal(mean_y, std_var_y, num_samples))

    return np.array(x_data), np.array(y_data)

def main():
    st.title("Gaussain Sample Generator")

    num_modes_red = st.sidebar.text_input("Number of modes blue", value=0)
    num_modes_blue = st.sidebar.text_input("Number of modes red", value=0)
    num_samples = st.sidebar.text_input("Number of samples", value=0)

    if st.sidebar.button("Generate"):
        red_x, red_y = generate_class_data(int(num_modes_red), int(num_samples))
        blue_x, blue_y = generate_class_data(int(num_modes_blue), int(num_samples))

        red_df = pd.DataFrame({"x": red_x, "y": red_y, "color": "#ff0000"})
        blue_df = pd.DataFrame({"x": blue_x, "y": blue_y, "color": "#0000ff"})

        data = pd.concat([red_df, blue_df])

        st.scatter_chart(data=data, x="x", y="y", color="color", width=700, height=500)


if __name__ == "__main__":
    main()