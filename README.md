
# README - Overseas Traveler Arrival Prediction in Shanghai Based on Machine Learning

## Overview

The project, developed for the 16th China College Students’ Computer Design Competition, focuses on predicting the flow of overseas travelers to Shanghai. It leverages machine learning techniques and historical tourism data to forecast future trends in the city's tourism industry.

## Project Components

### Frontend

- Developed using Vue3, TypeScript, DataV, and ECharts.
- Features an interactive dashboard for visualizing tourist data, including weather forecasts, map views, key data displays, and predictive analytics for tourist arrivals and hotel data.
- Implements dynamic screen adaptation and data refreshing.

### Backend

- Built using the Django framework.
- Offers a range of APIs for managing data related to non-mainland tourists, hotel statistics, weather forecasts, and country rates.
- Includes endpoints for administrative tasks, data retrieval, and updates.
- Utilizes Django's powerful features for database and account management, with changes reflected in real-time on the front end.

### Machine Learning Model

- Employs a Bi-directional LSTM (Long Short-Term Memory) model for data prediction.
- The model is evaluated and used for prediction, incorporating logic for single-step multiple predictions.

### Data Handling

- Data scraping from Shanghai Statistical Bureau for missing data post-February 2020 and post-2022.
- The dataset spans from January 2015 to April 2023, with various tables in the database each containing up to 100 records.
- Data tables include information about overseas tourists, hotel data, and non-mainland tourist data.

### System Deployment

- A Docker image for system deployment is prepared and can be loaded and run using Docker commands.
- Detailed instructions were provided for setting up Docker, installing Python and other dependencies, and configuring environment variables.

## API Introduction

- The APIs of this project can be found in the urls within the TravelServer folder.

- Below are the listed APIs:

  ```
  path("admin/", admin.site.urls),
  path("api/data/nmainland/all", api.views.api_nmainland_all),
  path('api/data/nmainland/sum/<int:year>', api.views.api_nmainland_sum_year),
  path('api/data/nmainland/per/<int:year>', api.views.api_nmainland_per_year),
  path('api/data/nmainland/sum/<int:year>/<int:month>', api.views.api_nmainland_sum_month),
  path('api/data/nmainland/per/<int:year>/<int:month>', api.views.api_nmainland_per_month),
  path("api/data/hotel/all", api.views.api_hotel_all),
  path("api/data/hotel/per", api.views.api_hotel_rate),
  path("api/data/weather", api.views.api_weather),
  path("api/data/country/rate", api.views.api_country_rate),
  path('', TemplateView.as_view(template_name='index.html'))
  ```

- **Admin API**: Opens the administrator interface for managing databases and accounts, reflecting changes directly on the frontend.

- **Frontend Integration**: The non-API link integrates frontend and backend, enabling Django to start both simultaneously.

### Data Type APIs

- APIs follow the format: `api/(return type)/(database table)/(operation)/(conditions)...`
- Operations include 'all', 'per', 'sum', for querying all data, year-over-year growth, and totals, respectively.
- Return values are in JSON format, with each API type returning specific data structures.

### Additional Information

- **Celery Commands**: (Windows specific; not needed for Linux)
  - `celery worker -A tasks --loglevel=info -P eventlet`
  - `celery -A tasks worker --loglevel=info -P eventlet`
- **Redis**: `./redis-server.exe redis.windows.conf`

## Required Packages

- django==3.1.7
- dmPython
- django_dmPython
- eventlet (latest version)
- celery (latest version)
- pandas
- redis
- django-pandas
- django-cors-headers
- numpy (bundled installation)
- scikit_learn

## Usage

- The system's main page displays future weather, key current, and predictive data, helping users understand future tourism trends.
- The backend management system allows user data maintenance and machine learning operations, including training duration settings, hyperparameter adjustments, and predictive data management.

## Contributing

Contributors can improve or extend the project by following standard open-source contribution guidelines, including forking the repository, creating feature branches, and submitting pull requests.
