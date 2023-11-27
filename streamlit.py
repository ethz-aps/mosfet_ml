import streamlit as st
from pycaret.classification import *
import numpy as np
from matplotlib import pyplot as plt


def main():
	inputs = [
		{'name':'TrapConc', 'low':200000000000., 'high':20000000000000., 'mean':10100000000000.0, 'format': '%.4e'},
		{'name':'Epi2Conc', 'low':1e+16, 'high':2.8e+16, 'mean':2.1547169811320756e+16, 'format': '%.4e'},
		{'name':'WellLeft', 'low':3.35, 'high':4.05, 'mean':3.558490566037736, 'format': '%.4f'},
		{'name':'nPlusShift', 'low':0.35, 'high':0.5, 'mean':0.3938679245283019, 'format': '%.4f'},
		{'name':'Kch', 'low':0.5, 'high':1.1, 'mean':0.8641509433962263, 'format': '%.4f'}
	]

	output = ['gm', 'RdsON_gm', 'Rch_gm', 'Rjfet_gm', 'Rdrift_gm', 'IdSat_max', 'IdSC800V_max', 'SCWT800V_us']


	with st.sidebar:
		st.markdown(
        f"""
            <style>
                [data-testid="stSidebar"] {{
                    background-image: url(https://aps.ee.ethz.ch/_jcr_content/orgbox/image.imageformat.logo.34196885.png);
                    background-repeat: no-repeat;
					background-size: 250px;
                }}
            </style>
            """,
        unsafe_allow_html=True)
		st.header("MOSFET Eyeballer\nModify to your liking:")

		params_in =[st.slider(d['name'], d['low'], d['high'], d['mean'], format=d['format']) for d in inputs]

		st.markdown("---")
		st.header("Plotting")
		x_options = [d['name'] for d in inputs]
		xax_label = st.selectbox('X-axis', set(x_options))
		yax_label = st.selectbox('Y-axis', set(output))
		plot_points = st.number_input('No. of plot points', 2, 200, 10)


	

	st.header("Output")
	output = ['gm', 'RdsON_gm', 'Rch_gm', 'Rjfet_gm', 'Rdrift_gm', 'IdSat_max', 'IdSC800V_max', 'SCWT800V_us']
	for target in output:
		cl = load_model(f'models/{target}')
		ret = cl.predict([params_in])
		st.markdown(f"#### {target}: {ret[0]:.4f}")

	st.markdown("---")

	cl = load_model(f'models/{yax_label}')
	res = [[idx, d] for idx, d in enumerate(inputs) if d['name'] == xax_label][0]

	xax = np.linspace(res[1]['low'], res[1]['high'], plot_points)

	yax = []
	for x in xax:
		model_input = params_in
		model_input[res[0]] = x
		y = cl.predict([model_input])
		yax.append(y)
	fig, ax = plt.subplots()
	ax.set_xlabel(xax_label)
	ax.set_ylabel(yax_label)
	ax.plot(xax, yax)
	st.pyplot(fig)


if __name__ == "__main__":
	st.set_page_config(
		page_title="MOSFET Eyeballer", page_icon=":chart_with_upwards_trend:"
	)
	main()


		