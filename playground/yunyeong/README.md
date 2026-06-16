## Jetson 실행 방법

이 스크립트는 기본적으로 Host의 `~/efficientat_ws` 경로가 존재한다고 가정한다.  
따라서 GitHub에서 받은 `playground/yunyeong/efficientat_ws` 폴더를 반드시 Host의 홈 디렉토리로 복사한 뒤 실행해야 한다.

```bash
cd ~

git clone -b playground https://github.com/26-DGU-CECD/EdgeAudioRecognition.git

rm -rf ~/efficientat_ws
cp -r ~/EdgeAudioRecognition/playground/yunyeong/efficientat_ws ~/efficientat_ws

cd ~/efficientat_ws
chmod +x run_wifi_bridge_finetuned_from_host.sh

./run_wifi_bridge_finetuned_from_host.sh plughw:2,0 8765
```
