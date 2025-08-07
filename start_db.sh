sudo kill -9 $(sudo lsof -t -i :19530) 2>/dev/null
sudo kill -9 $(sudo lsof -t -i :9091) 2>/dev/null
sudo kill -9 $(sudo lsof -t -i :9000) 2>/dev/null
sudo kill -9 $(sudo lsof -t -i :9001) 2>/dev/null
docker container prune -f
docker network prune -f
docker network rm milvus
docker compose up -d