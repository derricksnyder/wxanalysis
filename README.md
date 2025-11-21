This is a great set of scripts\! You have a solid portfolio here covering **Model Analysis** (HRRR), **Point Analysis** (Soundings), and **Observational Analysis** (MRMS Radar).

Below is a professional `README.md` file tailored specifically to the code you just uploaded. It explains what the tools do, what libraries are needed, and how to run them.

### How to add this to GitHub

1.  Go to your repository main page.
2.  Click **Add file** \> **Create new file**.
3.  Name the file: `README.md`
4.  Paste the content below into the text box.
5.  Click **Commit changes**.

-----

# Meteorological Analysis Tools

A collection of Python tools designed for analyzing severe weather potential using HRRR model data and MRMS observational radar data. These tools utilize **Herbie** for data retrieval and **MetPy** for meteorological calculations.

## 📂 Included Tools

### 1\. Severe Weather Analyzer (`severe_weather_analyzer.py`)

An interactive tool for mapping spatial severe weather parameters from the HRRR model.

  * **Data Source:** HRRR Model (via Herbie).
  * **Features:**
      * **Layered Plotting:** Overlay vector data (Wind Barbs) on top of scalar data (e.g., Dewpoint, CAPE).
      * **Derived Indices:** Automatically calculates complex parameters like **Supercell Composite (SCP)**, **Significant Tornado Parameter (STP)**, and **Bulk Shear**.
      * **Parallel Processing:** Uses multiprocessing to render time-series images rapidly.
      * **GIF Generation:** Automatically stitches output images into animations.
      * **Domains:** Pre-configured for Regional and NWS Paducah (PAH) zoomed views.

### 2\. Sounding Analyzer (`sounding_analyzer.py`)

Generates vertical profile soundings (Skew-T Log-P) and Hodographs for any specific latitude/longitude.

  * **Data Source:** HRRR Model (via Herbie).
  * **Modes:**
      * **Convective Mode:** Highlights Instability (CAPE/CIN), LCL/LFC heights, and Storm Relative Helicity (SRH).
      * **Winter Mode:** Highlights dendritic growth zones (DGZ) and freezing levels.
  * **Visualization:** Features a clean, professional layout ("SHARPy-style") with a dedicated Hodograph inset and calculated metric tables.

### 3\. MRMS Radar Analyzer (`mrms_analyzer.py`)

Downloads and plots high-resolution observational radar products from the Multi-Radar Multi-Sensor (MRMS) system.

  * **Data Source:** Iowa State IEM Archives (GRIB2).
  * **Products:**
      * **Reflectivity:** Seamless Hybrid Scan Reflectivity.
      * **Rotation Tracks:** 30-min and 1440-min accumulated rotation.
      * **QPE:** Quantitative Precipitation Estimation (1hr, 24hr, 72hr) with custom color tables.
  * **Features:** Handles complex GRIB2 compression and multiprocessing downloads for high-speed retrieval.

-----

## 🛠️ Installation & Requirements

To make setup easy, this project includes an `environment.yml` file. This will automatically install Python and all required libraries (Herbie, MetPy, Cartopy, etc.) in one step.

**Prerequisites:** You need to have [Anaconda](https://www.anaconda.com/) or [Miniforge](https://github.com/conda-forge/miniforge) installed.

1.  **Clone (download) this repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate severe_wx_env
    ```
    *(Note: You must activate this environment every time you want to run the scripts)*

## 🚀 Usage

All scripts are interactive CLI (Command Line Interface) tools. They will prompt you for dates, times, and locations.

### Running the Severe Weather Analyzer

```bash
python severe_weather_analyzer.py
```

*Follow the prompts to select start/end times, plot types (e.g., SBCAPE, STP), and domain.*

### Running the Sounding Analyzer

```bash
python sounding_analyzer.py
```

*Enter a specific Latitude/Longitude (e.g., 37.08, -88.6) and the analysis mode (Winter/Convective).*

### Running the MRMS Analyzer

```bash
python mrms_analyzer.py
```

*Select products (Reflectivity, Rotation, QPE) and the time step for animations.*

-----

## 🗺️ Configuration

The scripts currently default to a domain centered on the **Ohio Valley / NWS Paducah** region. You can adjust the bounding boxes in the code configuration sections:

```python
# Example in severe_weather_analyzer.py
REGIONAL_DOMAIN = {
    'name': 'Regional (Default)',
    'west': -92.2,
    'east': -85.0,
    'south': 34.68,
    'north': 39.48
}
```

## 📄 License

This project is open source. Feel free to modify and adapt the scripts for your own forecast areas\!
