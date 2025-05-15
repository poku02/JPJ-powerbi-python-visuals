# JPJ-powerbi-python-visuals
This Python script allows you to automatically analyse and visualise data.

Project Dashboard for Power BI
This repository contains a Python script that creates a beautiful, interactive dashboard with a gold and wine color theme for use in Power BI.
Overview
The project_dashboard.py script generates a comprehensive dashboard with multiple visualizations:
1. Key Performance Indicators (KPIs) 2. Time Series Analysis
3. Distribution Analysis (Pie Chart) 4. Comparative Analysis (Bar Chart) 5. Correlation Analysis (Scatter Plot) 6. Correlation Heatmap
The dashboard automatically adapts to your data and provides fallback visual- izations if certain data types are not available.
Requirements
To use this dashboard in Power BI, you need the following Python packages: - matplotlib - pandas - numpy - seaborn - datetime
How to Use in Power BI
Step 1: Set Up Python in Power BI
1. Install Python on your computer if you haven’t already.
2. Install the required packages using pip:
     pip install matplotlib pandas numpy seaborn
3. In Power BI Desktop, go to File > Options and settings > Options.
4. Navigate to Python scripting in the left menu.
5. Set the path to your Python installation.
Step 2: Import Your Data into Power BI
1. Import your data into Power BI using any of the available connectors.
2. Make sure your data is clean and properly formatted.
Step 3: Add the Python Visual
1. In the Power BI Desktop, click on the Python visual icon in the Visual- izations pane.
2. A Python script editor will appear at the bottom of the screen. 1

3. Copy the entire content of project_dashboard.py into this editor.
4. Select the fields from your data that you want to include in the dashboard
by dragging them to the “Values” section of the Python visual.
Step 4: Run the Script
1. Power BI will automatically run the script when you’ve added the neces- sary fields.
2. The dashboard will be generated based on your data.
3. If your data contains date columns, they will be automatically detected
for time series analysis.
4. Numeric columns will be used for KPIs and correlation analysis.
5. Categorical columns will be used for pie charts and bar charts.
Tips for Best Results
1. Date Columns: Ensure date columns are properly formatted as dates in Power BI before using them in the Python visual.
2. Categorical Data: For best results with pie charts and bar charts, in- clude categorical columns with a reasonable number of categories (5-10 is ideal).
3. Numeric Data: Include multiple numeric columns for the correlation heatmap to be most effective.
4. Visual Size: Adjust the size of the Python visual in Power BI to be large enough to display all elements of the dashboard clearly (recommended size: 16x10 or larger).
5. Refresh: If you update your data, you may need to refresh the Python visual by clicking on it and then clicking the “Refresh” button.
Troubleshooting
If the dashboard doesn’t appear correctly:
1. Check the Python script editor for any error messages.
2. Verify that all required packages are installed.
3. Make sure your data contains appropriate column types (dates, numbers,
categories).
4. Try increasing the size of the Python visual in Power BI.
5. If using very large datasets, consider filtering or aggregating the data
before using it with the Python visual.
Customization
You can customize the dashboard by modifying the project_dashboard.py file:
• Change the color scheme by modifying the GOLD, WINE, LIGHT_GOLD, and DARK_WINE variables.
2

• Adjust the layout by modifying the fig.add_gridspec(3, 3) line and the subsequent subplot definitions.
• Add or remove visualizations by adding or removing the corresponding code sections.
License
This project is available for free use and modification.