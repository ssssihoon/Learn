# Find, Singleton, Coroutine

# Find 함수

- 특정 이름을 가진 게임 오브젝트를 찾아 반환한다. (첫 번째 게임 오브젝트를 반환)
    - 없다면 null반환

```csharp
// 특정 이름을 가진 게임 오브젝트 찾기
GameObject myObject = GameObject.Find("MyObjectName");

if (myObject != null)
{
    // 게임 오브젝트를 찾았을 때의 동작
    // ...
}
```

## GetComponent<> 메서드

- 게임 오브젝트에서 특정 컴포넌트의 인스턴스를 가져온다.  제네릭 안에 컴포넌트 타입을 지정
    - 해당 컴포넌트가 없으면 null 반환

# Singleton

- 동적인 데이터는 싱글톤으로 관리하는 것이 효율적이다. ; 반면 맵 정보와 같이 일반적으로 변경되지 않는 정적인 데이터는 싱글톤을 사용하지 않는다.

**플레이어 상태 매니저:**

- 플레이어의 현재 상태를 추적하고 관리할 수 있습니다. 예를 들어 플레이어의 위치, 상태, 경험치 등을 다룰 수 있습니다.

**사운드 매니저:**

- 게임 내에서 발생하는 사운드 이펙트나 배경 음악을 관리할 수 있습니다.

```csharp
using UnityEngine;

public class SoundManager : MonoBehaviour
{
    // 싱글톤 인스턴스
    private static SoundManager instance;

    // 한국 맵에 대한 배경음악
    public AudioClip koreaBackgroundMusic;

    // 미국 맵에 대한 배경음악
    public AudioClip usaBackgroundMusic;

    // 다른 스크립트에서 이 인스턴스에 접근하기 위한 프로퍼티
    public static SoundManager Instance
    {
        get
        {
            // 만약 인스턴스가 없다면 새로운 게임 오브젝트를 생성하고 그에게 SoundManager 스크립트를 추가합니다.
            if (instance == null)
            {
                GameObject soundManagerObject = new GameObject("SoundManager");
                instance = soundManagerObject.AddComponent<SoundManager>();
            }

            // 인스턴스를 반환합니다.
            return instance;
        }
    }

    // 배경음악을 재생하는 메서드
    public void PlayBackgroundMusic(string mapName)
    {
        AudioSource audioSource = GetComponent<AudioSource>();

        // 맵 이름에 따라 다른 배경음악을 선택하여 재생
        if (mapName == "Korea")
        {
            audioSource.clip = koreaBackgroundMusic;
        }
        else if (mapName == "USA")
        {
            audioSource.clip = usaBackgroundMusic;
        }

        // 재생
        audioSource.Play();
    }

    // 기타 사운드 이펙트 재생 등의 메서드들을 추가할 수 있습니다.
}
```

# Coroutine

[https://sharp2studio.tistory.com/14](https://sharp2studio.tistory.com/14)

**시간의 경과에 따른** 명령을 주고싶을 때 사용하게 되는 문법이다.

Update문으로 프레임마다 적용하여 시간을 계산하는 방법이 있지만 코루틴을 사용하면 훨씬 간단해진다.

```csharp
private void Start()
    {       
        StartCoroutine("CoroutineName");
        Invoke("CoroutineStop", 3); //시작 후 3초뒤에 CoroutineStop이라는 함수를 호출함.
    }
    IEnumerator CoroutineName()
    {
        int counter=0;

        while (true)
        {
            Debug.Log(counter);
            counter++;
            yield return new WaitForSeconds(1);
        }
    }
    void CoroutineStop() //시작 후 3초뒤에 호출 될 함수
    {
        Debug.Log("코루틴 종료");
        StopCoroutine("CoroutineName");     
    }
```
![][https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ezVZE-vJqcvj7Frsp_H68w.gif](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ezVZE-vJqcvj7Frsp_H68w.gif)
