# GitHub Actions Setup Guide

This guide explains how to set up automated testing and CI/CD for your vCenter DRS Compliance Dashboard using GitHub Actions.

## 🚀 **What GitHub Actions Does**

GitHub Actions automatically runs tests and checks on every commit and pull request to ensure:

- ✅ **Code Quality**: Linting, type checking, and formatting
- ✅ **Functionality**: Unit tests and integration tests
- ✅ **Security**: Security vulnerability scanning
- ✅ **Build Verification**: Package building and validation
- ✅ **Deployment Scripts**: Validation of deployment and cron scripts

## 📁 **Workflow Files**

### **1. Main CI Pipeline** (`.github/workflows/ci.yml`)

**Triggers**: Every push to `main`/`develop` and all pull requests

**Jobs**:
- **Test**: Runs on Python 3.8-3.12 with MySQL 8.0
- **Security**: Security scanning with Bandit and Safety
- **Build**: Package building and validation

**What it tests**:
- Code linting (flake8)
- Type checking (mypy)
- Unit tests (pytest with coverage)
- Security vulnerabilities
- Package building

### **2. Cron Script Testing** (`.github/workflows/test-cron.yml`)

**Triggers**: When cron/deployment scripts are modified

**Jobs**:
- **test-cron-scripts**: Tests the automated scripts
- **test-deployment**: Validates deployment configuration

**What it tests**:
- Script syntax validation
- Import testing
- Service file validation
- Deployment script logic

## 🛠️ **Setup Instructions**

### **Step 1: Enable GitHub Actions**

1. **Push your code to GitHub** (if not already done)
2. **Go to your repository** on GitHub
3. **Click "Actions" tab**
4. **GitHub will automatically detect** the workflow files

### **Step 2: Configure Repository Settings**

1. **Go to Settings → Actions → General**
2. **Enable "Allow all actions and reusable workflows"**
3. **Set "Workflow permissions" to "Read and write permissions"**

### **Step 3: Add Repository Secrets (Optional)**

For enhanced security, add these secrets in **Settings → Secrets and variables → Actions**:

```
VCENTER_HOST=your-vcenter-server.com
VCENTER_USERNAME=your-username
VCENTER_PASSWORD=your-password
DB_HOST=your-db-host
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_DATABASE=your-db-name
```

### **Step 4: Set Up Code Coverage (Optional)**

1. **Sign up for Codecov** (https://codecov.io)
2. **Connect your GitHub repository**
3. **Add the Codecov token** as a repository secret named `CODECOV_TOKEN`

## 🧪 **Understanding the Tests**

### **Unit Tests**

Located in `vcenter_drs/tests/`:

- **`test_vcenter_client.py`**: Tests vCenter API client
- **`test_metrics_db.py`**: Tests database operations
- **`conftest.py`**: Common test fixtures and configuration

### **Test Fixtures**

The `conftest.py` file provides reusable test components:

```python
@pytest.fixture
def mock_vcenter_connection():
    """Mock vCenter connection for testing."""
    
@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    
@pytest.fixture
def sample_vm_data():
    """Sample VM data for testing."""
```

### **Running Tests Locally**

```bash
# Install test dependencies
pip install -r vcenter_drs/requirements.txt[dev]

# Run all tests
cd vcenter_drs
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_vcenter_client.py -v
```

## 📊 **Understanding Test Results**

### **GitHub Actions Dashboard**

1. **Go to Actions tab** in your repository
2. **Click on a workflow run** to see detailed results
3. **Expand job steps** to see individual test results

### **Test Status Indicators**

- ✅ **Green**: All tests passed
- ❌ **Red**: Tests failed
- ⚠️ **Yellow**: Tests passed with warnings
- 🔄 **Blue**: Tests are running

### **Coverage Reports**

If Codecov is set up:
- **Coverage percentage** is shown in pull requests
- **Detailed coverage reports** are available on Codecov
- **Coverage trends** are tracked over time

## 🔧 **Customizing the Workflows**

### **Adding New Tests**

1. **Create test files** in `vcenter_drs/tests/`
2. **Follow naming convention**: `test_*.py`
3. **Use pytest fixtures** from `conftest.py`
4. **Push to trigger** automatic testing

### **Modifying Test Matrix**

Edit `.github/workflows/ci.yml`:

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10, 3.11, 3.12]  # Add/remove versions
    mysql-version: ['8.0', '5.7']  # Add more MySQL versions
```

### **Adding New Checks**

Add new steps to the workflow:

```yaml
- name: Run custom check
  run: |
    python custom_check.py
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Tests Failing Locally but Passing in CI**

- **Check Python version**: Ensure you're using the same version
- **Check dependencies**: Install all dev dependencies
- **Check environment**: Set `TESTING=true` environment variable

#### **2. Import Errors in Tests**

- **Check PYTHONPATH**: Ensure the module is in the path
- **Check virtual environment**: Activate the correct environment
- **Check relative imports**: Use absolute imports in tests

#### **3. Database Connection Issues**

- **Check MySQL service**: Ensure MySQL is running
- **Check credentials**: Verify test credentials are correct
- **Check port**: Ensure port 3306 is available

### **Debugging Workflows**

1. **Enable debug logging**: Add `ACTIONS_STEP_DEBUG: true` as a repository secret
2. **Check workflow logs**: Expand failed steps to see detailed error messages
3. **Run tests locally**: Reproduce issues locally first

## 📈 **Best Practices**

### **Writing Tests**

1. **Use descriptive test names**: `test_connect_success()` not `test_1()`
2. **Test one thing per test**: Each test should verify one specific behavior
3. **Use mocks for external dependencies**: Don't rely on real vCenter/database
4. **Clean up after tests**: Use fixtures for setup/teardown

### **Maintaining Workflows**

1. **Keep workflows fast**: Use caching and parallel jobs
2. **Fail fast**: Put quick checks first
3. **Use specific versions**: Pin dependency versions
4. **Document changes**: Update this guide when modifying workflows

### **Security Considerations**

1. **Never commit secrets**: Use repository secrets for sensitive data
2. **Use least privilege**: Give workflows minimal permissions
3. **Scan for vulnerabilities**: Run security checks regularly
4. **Keep dependencies updated**: Regular security updates

## 🎯 **Next Steps**

1. **Push the workflow files** to your repository
2. **Monitor the first run** to ensure everything works
3. **Add more tests** for uncovered functionality
4. **Set up branch protection** to require passing tests
5. **Configure automated deployments** (optional)

## 📚 **Additional Resources**

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)

## 🤝 **Getting Help**

If you encounter issues:

1. **Check the workflow logs** for detailed error messages
2. **Search GitHub Issues** for similar problems
3. **Create a new issue** with detailed information
4. **Ask in the community** forums 