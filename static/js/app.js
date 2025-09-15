// Traffic Prediction System - Frontend JavaScript
class TrafficDashboard {
    constructor() {
        this.currentLocation = null;
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadLocations();
        await this.loadSystemStatus();
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('locationSelect').addEventListener('change', (e) => {
            this.currentLocation = e.target.value;
            if (this.currentLocation) {
                this.loadData();
            }
        });
    }

    async loadLocations() {
        try {
            const response = await fetch('/api/locations');
            const locations = await response.json();
            
            const select = document.getElementById('locationSelect');
            select.innerHTML = '<option value="">Select a location...</option>';
            
            locations.forEach(location => {
                const option = document.createElement('option');
                option.value = location.location_id;
                option.textContent = location.location_name;
                select.appendChild(option);
            });
        } catch (error) {
            this.showError('Failed to load locations: ' + error.message);
        }
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            const statusHtml = `
                <div class="stat-card">
                    <div class="stat-value">${status.locations_available}</div>
                    <div class="stat-label">Locations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${status.trained_models}</div>
                    <div class="stat-label">Trained Models</div>
                </div>
            `;
            
            document.getElementById('systemStatus').innerHTML = statusHtml;
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }

    async loadData() {
        if (!this.currentLocation) {
            this.showError('Please select a location first');
            return;
        }

        this.showLoading('overview');
        
        try {
            const days = document.getElementById('historicalDays').value;
            const response = await fetch(`/api/historical/${this.currentLocation}?days=${days}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.displayHistoricalData(data);
            await this.loadAnalytics();
            
        } catch (error) {
            this.showError('Failed to load data: ' + error.message);
        }
    }

    displayHistoricalData(data) {
        // Update overview stats
        const totalTraffic = data.data.reduce((sum, item) => sum + item.traffic_volume, 0);
        const avgTraffic = Math.round(totalTraffic / data.data.length);
        const maxTraffic = Math.max(...data.data.map(item => item.traffic_volume));
        const minTraffic = Math.min(...data.data.map(item => item.traffic_volume));

        const statsHtml = `
            <div class="stat-card">
                <div class="stat-value">${avgTraffic}</div>
                <div class="stat-label">Avg Traffic/Hour</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${maxTraffic}</div>
                <div class="stat-label">Peak Traffic</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${minTraffic}</div>
                <div class="stat-label">Min Traffic</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.total_records}</div>
                <div class="stat-label">Data Points</div>
            </div>
        `;
        
        document.getElementById('overviewStats').innerHTML = statsHtml;

        // Create historical chart
        this.createHistoricalChart(data.data);
    }

    createHistoricalChart(data) {
        const ctx = document.getElementById('historicalChart').getContext('2d');
        
        if (this.charts.historical) {
            this.charts.historical.destroy();
        }

        const labels = data.map(item => new Date(item.timestamp).toLocaleDateString());
        const trafficData = data.map(item => item.traffic_volume);

        this.charts.historical = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Traffic Volume',
                    data: trafficData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Historical Traffic Data'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Vehicle Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    async makePrediction() {
        if (!this.currentLocation) {
            this.showError('Please select a location first');
            return;
        }

        this.showLoading('predictions');
        
        try {
            const hours = document.getElementById('hoursAhead').value;
            const response = await fetch(`/api/predict/${this.currentLocation}?hours=${hours}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.displayPredictions(data);
            this.switchTab('predictions');
            
        } catch (error) {
            this.showError('Failed to make prediction: ' + error.message);
        }
    }

    displayPredictions(data) {
        // Update prediction stats
        const predictions = data.predictions.map(p => p.predicted_traffic);
        const avgPredicted = Math.round(predictions.reduce((a, b) => a + b, 0) / predictions.length);
        const maxPredicted = Math.max(...predictions);
        const minPredicted = Math.min(...predictions);

        const statsHtml = `
            <div class="stat-card">
                <div class="stat-value">${avgPredicted}</div>
                <div class="stat-label">Avg Predicted</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${maxPredicted}</div>
                <div class="stat-label">Peak Predicted</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${minPredicted}</div>
                <div class="stat-label">Min Predicted</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.model_metrics.test_r2.toFixed(3)}</div>
                <div class="stat-label">Model R² Score</div>
            </div>
        `;
        
        document.getElementById('predictionStats').innerHTML = statsHtml;

        // Create prediction chart
        this.createPredictionChart(data.predictions);
        
        // Create prediction list
        this.createPredictionList(data.predictions);
    }

    createPredictionChart(predictions) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        if (this.charts.prediction) {
            this.charts.prediction.destroy();
        }

        const labels = predictions.map(item => {
            const date = new Date(item.timestamp);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        });
        const trafficData = predictions.map(item => item.predicted_traffic);

        this.charts.prediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Traffic',
                    data: trafficData,
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Traffic Predictions'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Predicted Vehicle Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    createPredictionList(predictions) {
        const listHtml = predictions.map(item => {
            const date = new Date(item.timestamp);
            const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            return `
                <div class="prediction-item">
                    <div class="prediction-time">${timeStr}</div>
                    <div class="prediction-value">${item.predicted_traffic} vehicles</div>
                </div>
            `;
        }).join('');
        
        document.getElementById('predictionList').innerHTML = listHtml;
    }

    async loadAnalytics() {
        if (!this.currentLocation) return;

        try {
            const response = await fetch(`/api/analytics/${this.currentLocation}?days=30`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.displayAnalytics(data.analytics);
            
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    displayAnalytics(analytics) {
        const ctx = document.getElementById('analyticsChart').getContext('2d');
        
        if (this.charts.analytics) {
            this.charts.analytics.destroy();
        }

        // Create hourly pattern chart
        const hours = Object.keys(analytics.hourly_pattern).map(h => `${h}:00`);
        const hourlyTraffic = Object.values(analytics.hourly_pattern);

        this.charts.analytics = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Average Traffic by Hour',
                    data: hourlyTraffic,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Traffic Patterns by Hour of Day'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Average Vehicle Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        }
                    }
                }
            }
        });
    }

    async trainModel() {
        if (!this.currentLocation) {
            this.showError('Please select a location first');
            return;
        }

        this.showLoading('overview');
        
        try {
            const response = await fetch(`/api/train/${this.currentLocation}`, {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.showSuccess('Model trained successfully! R² Score: ' + data.metrics.test_r2.toFixed(3));
            await this.loadSystemStatus();
            
        } catch (error) {
            this.showError('Failed to train model: ' + error.message);
        }
    }

    switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected tab content
        document.getElementById(tabName).classList.add('active');
        
        // Add active class to selected tab
        event.target.classList.add('active');
    }

    showLoading(containerId) {
        const container = document.getElementById(containerId);
        const loadingHtml = `
            <div class="loading">
                <i class="fas fa-spinner"></i>
                Loading...
            </div>
        `;
        container.innerHTML = loadingHtml;
    }

    showError(message) {
        const errorHtml = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </div>
        `;
        
        // Show error in the active tab
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab) {
            activeTab.insertAdjacentHTML('afterbegin', errorHtml);
            
            // Remove error after 5 seconds
            setTimeout(() => {
                const errorDiv = activeTab.querySelector('.error');
                if (errorDiv) {
                    errorDiv.remove();
                }
            }, 5000);
        }
    }

    showSuccess(message) {
        const successHtml = `
            <div class="success">
                <i class="fas fa-check-circle"></i>
                ${message}
            </div>
        `;
        
        // Show success in the active tab
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab) {
            activeTab.insertAdjacentHTML('afterbegin', successHtml);
            
            // Remove success message after 5 seconds
            setTimeout(() => {
                const successDiv = activeTab.querySelector('.success');
                if (successDiv) {
                    successDiv.remove();
                }
            }, 5000);
        }
    }
}

// Global functions for HTML onclick events
function loadData() {
    dashboard.loadData();
}

function makePrediction() {
    dashboard.makePrediction();
}

function trainModel() {
    dashboard.trainModel();
}

function switchTab(tabName) {
    dashboard.switchTab(tabName);
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TrafficDashboard();
});
