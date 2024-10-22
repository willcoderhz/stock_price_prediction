# Stock Price Prediction Django App

This project is a Django-based application that fetches historical stock data from the Alpha Vantage API and stores it in a PostgreSQL database. The application is Dockerized for easy deployment and has a CI/CD pipeline using GitHub Actions, with the deployment target set to AWS Elastic Beanstalk.

## Features

- Fetch stock data using Alpha Vantage API.
- Store and manage stock data in a PostgreSQL database.
- Predict stock prices using a pre-trained model (e.g., linear regression).
- Dockerized setup for local development and production deployment.
- CI/CD pipeline using GitHub Actions.
- Deployed to AWS Elastic Beanstalk.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Locally](#setup-locally)
- [Environment Variables](#environment-variables)
- [Running the App](#running-the-app)
- [Database Migrations](#database-migrations)
- [Testing](#testing)
- [Deployment to AWS Elastic Beanstalk](#deployment-to-aws-elastic-beanstalk)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- **Django**: Web framework for building the backend.
- **PostgreSQL**: Database for storing stock data.
- **Docker**: For containerizing the application.
- **GitHub Actions**: For setting up CI/CD pipeline.
- **AWS Elastic Beanstalk**: For deployment.
- **Alpha Vantage API**: To fetch stock data.

## Setup Locally

### Prerequisites

Make sure you have the following installed:

- **Docker**: [Get Docker](https://www.docker.com/get-started)
- **Git**: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **PostgreSQL**: Installed locally or you can use a cloud-based PostgreSQL service like AWS RDS.
- **Alpha Vantage API Key**: [Get your API key here](https://www.alphavantage.co/).

### Cloning the Repository

```bash
git clone https://github.com/yourusername/stock_price_prediction.git
cd stock_price_prediction
```

### Environment Variables
Create a .env file in the root of the project and add the following environment variables:

- **Alpha Vantage API**
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

- **PostgreSQL**
POSTGRES_DB=your_db_name
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=your_rds_or_local_db_host
POSTGRES_PORT=5432

- **Django Secret Key**
DJANGO_SECRET_KEY=your_django_secret_key

*You can get your Alpha Vantage API key from Alpha Vantage.*

### Running the App

Build and run the application with Docker
Make sure Docker is running, then run the following commands:

```bash
docker-compose up --build
```

## Access the application

Once the containers are up, visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## Stopping the App

To stop the running containers:

```bash
docker-compose down
```

## Deployment to AWS Elastic Beanstalk

This project is configured to use GitHub Actions for CI/CD and to deploy automatically to AWS Elastic Beanstalk.

### Deployment Steps:

1. **Configure AWS Credentials**: Add your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as secrets in your GitHub repository under Settings -> Secrets and variables -> Actions -> New repository secret.

2. **Push to Main**: Any push to the main branch will trigger a GitHub Actions workflow that will:
   - Build the Docker image.
   - Push the image to DockerHub.
   - Deploy the application to AWS Elastic Beanstalk.

3. **Access the Application**: Once deployed, you can access your application via the Elastic Beanstalk-provided URL (e.g., `http://your-app-name.us-east-2.elasticbeanstalk.com`).

### Manual Deployment

You can also manually deploy using the Elastic Beanstalk console by uploading a new Docker image.

### Monitoring

You can monitor your deployed environment's health and logs using the AWS Elastic Beanstalk console.

### Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**: Fork it

2. **Create your feature branch**:

   ```bash
   git checkout -b feature/YourFeature
   ```

### Commit your changes:
  ```bash
    git commit -am 'Add some feature'
  ```

### Push to the branch:

 ```bash
git push origin feature/YourFeature
  ```

Create a new Pull Request

### License
This project is licensed under the MIT License. See the LICENSE file for details.
