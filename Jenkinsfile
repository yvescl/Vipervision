// Needs an agent with Docker. Builds the image (which verifies the hashed
// requirements install) and smoke-tests it; nothing is pushed or deployed.
pipeline {
    agent any
    options {
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }
    environment {
        IMAGE = "vipervision:ci-${env.BUILD_NUMBER}"
        CONTAINER = "vipervision-ci-${env.BUILD_NUMBER}"
    }
    stages {
        stage('Build image') {
            steps {
                sh 'docker build -t "$IMAGE" .'
            }
        }
        stage('Smoke test') {
            steps {
                sh '''
                    docker run --rm "$IMAGE" python -c "import app, cast, clip, detector, storage"
                    docker run -d --name "$CONTAINER" \
                        -e VIPERVISION_PASSWORD=ci -e SECRET_KEY=ci \
                        -e RTSP_URL= -e MINIO_ENDPOINT= "$IMAGE"
                    for i in $(seq 1 20); do
                        if docker exec "$CONTAINER" python -c "import urllib.request as u; assert u.urlopen('http://localhost:5000/login').status == 200"; then
                            exit 0
                        fi
                        sleep 1
                    done
                    docker logs "$CONTAINER"
                    exit 1
                '''
            }
        }
    }
    post {
        always {
            sh 'docker rm -f "$CONTAINER" || true'
            sh 'docker rmi "$IMAGE" || true'
        }
    }
}
