# Audio Component

- Audio Clip
- Audio Source
- Audio Listener

## Audio Clip

- MP3와 같은 에셋

## Audio Source

- 오디오 소스(Audio Source) 는 씬에서 오디오 클립을 재생한다.

<img width="1436" alt="Untitled" src="https://github.com/ssssihoon/CodingTest_Algorithm/assets/127017020/ade12d57-4326-4d0c-890f-8ea16e05e1ab">


### property

| 프로퍼티 | 설명 |
| --- | --- |
| Audio Clip | 재생될 사운드 클립에 대한 레퍼런스입니다. |
| Output | 기본적으로 클립은 씬의 오디오 리스너에 직접 출력됩니다. 이 프로퍼티를 사용하면 클립을 오디오 믹서에 출력합니다. |
| Mute | 옵션을 활성화하면 사운드가 음소거된 상태로 재생이 됩니다. |
| Spatialize | 오디오 소스에 대한 커스텀 공간화를 활성화하거나 비활성화합니다.이 프로퍼티는 오디오 스페이셜라이저 SDK를 설치하고 프로젝트의 전역 audio 설정에서 선택한 경우에만 사용할 수 있습니다. |
| Spatialize Post Effect | 오디오 소스에 연결된 다른 효과 앞뒤에 커스텀 공간화를 적용할지 결정합니다.이 프로퍼티를 활성화하면 오디오 소스에 연결된 다른 효과 뒤에 커스텀 공간화를 적용합니다.이 프로퍼티는 오디오 소스에 대해 Spatialize 프로퍼티를 활성화한 경우에만 사용할 수 있습니다. |
| Bypass Effects | 오디오 소스에 적용된 필터 효과를 즉시 “바이패스”합니다. 모든 효과를 쉽게 켜고 끌 수 있는 옵션입니다. |
| Bypass Listener Effects | 모든 리스너 효과를 즉시 켜거나 끄는 옵션입니다. |
| Bypass Reverb Zones | 모든 리버브 존을 즉시 켜거나 끄는 옵션입니다. |
| Play On Awake | 옵션을 활성화하면 씬이 실행되는 시점에 사운드 재생이 시작됩니다. 이 옵션을 비활성화하면 스크립팅에서 Play() 명령을 사용하여 사운드 재생을 시작해야 합니다. |
| Loop | 옵션을 활성화하면 재생이 끝날 때 Audio Clip 루프가 생성됩니다. |
| Priority | 씬 안에 공존하는 모든 오디오 소스 중 현재 오디오 소스의 우선 순위를 결정합니다(우선 순위: 0 = 최우선 순위, 256 = 최하 순위, 디폴트값 = 128). 순위가 잘못 바뀌는 경우를 막기 위해 음악 트랙은 0으로 설정해야 합니다. |
| Volume | Audio Listener__로부터 1월드 유닛(1미터) 거리에서 소리가 얼마나 크게 들리는지를 정의합니다. |
| Stereo Pan | 2D 사운드의 스테레오 필드 포지션을 설정합니다. |
| Spatial Blend | 3D 엔진이 오디오 소스에 미치는 효과의 정도를 설정합니다. |
| Reverb Zone Mix | 리버브 존으로 보내지는 출력 신호의 양을 설정합니다. 이 양은 0–1 범위에서는 선형적이며, 1–1.1 범위에서는 10dB 증폭이 가능하여 근거리 필드나 원거리 사운드 효과를 내는 데 유용합니다. |
| 3D Sound Settings | 설정은 공간 블렌드 파라미터에 비례해 적용됩니다. |
| Doppler Level | 현재 오디오 소스에 적용될 도플러 이펙트의 정도를 결정합니다(0으로 설정하면 아무 효과도 적용되지 않음). |
| Spread | 스피커 공백에서 3D 스테레오 사운드 또는 멀티채널 사운드로의 스프레드 각도를 설정합니다. |
| Min Distance | 사운드는 최소거리 내에서 가능한 한 최대 음량을 유지하고, 최소거리를 벗어나면 감쇠됩니다. 3d 세계에서 사운드를 ‘크게’ 만들기 위해서는 해당 사운드의 최소거리를 증가시키고, 사운드를 ‘작게’ 하려면 최소거리를 감소시키면 됩니다. |
| Max Distance | 사운드 감쇠가 더 이상 일어나지 않는 거리를 의미합니다. 이 포인트를 넘어서면 사운드는 리스너로부터 최대거리 유닛 위치의 영역으로 유지되며 더 이상 감쇠되지 않습니다. |
| Rolloff Mode | 사운드가 페이드되는 속도를 나타냅니다. 값이 클수록 리스너 위치가 더 가까워져야 사운드가 들립니다(그래프에 의해 결정됩니다). |
| - Logarithmic Rolloff | 오디오 소스에 가까우면 사운드가 크지만 오브젝트로부터 멀어지면 상당히 빠른 속도로 사운드가 작아집니다. |
| - Linear Rolloff | 오디오 소스로부터 멀어질수록 사운드가 점점 작아지게 됩니다. |
| - Custom Rolloff | 오디오 소스로부터의 사운드는 롤오프 그래프에 설정한 대로 작동합니다. |

## Audio Listener

- 씬 내 단일 컴포넌트이며 플레이어에게 있어 가상 귀의 역할을 함.

[https://gmls.tistory.com/74](https://gmls.tistory.com/74)

[https://docs.unity3d.com/kr/2022.3/Manual/class-AudioSource.html](https://docs.unity3d.com/kr/2022.3/Manual/class-AudioSource.html)

# UGUI

- 게임 및 응용 프로그램의 런타임 UI를 개발하는 데 사용할 수 있는 오래된 GameObject 기반 UI 시스템이다.
- Unity UI에서 구성요소 및 게임 뷰를 사용하여 사용자 인터페이스를 정렬, 배치 및 스타일링. 고급 렌더링 및 텍스트 기능을 지원

<img width="204" alt="Untitled 1" src="https://github.com/ssssihoon/CodingTest_Algorithm/assets/127017020/c0a557a7-f687-45b4-b200-9ba48ea4c95f">


[https://hkn10004.tistory.com/34](https://hkn10004.tistory.com/34)

# **Canvas & EventSystem**

## Canvas

- 캔버스는 렌더링을 관리하는 컴포넌트이다.
- 캔버스 영역은 씬 뷰에서 사각형으로 나타나므로 매번 게임 뷰가 보이게 하지 않고도 UI 요소를 배치하기 용이하다.

## EventSystem

- 이벤트 시스템은 키보드, 마우스, 터치, 커스텀 입력 등 입력 기반 애플리케이션의 오브젝트에 이벤트를 전송하는 방법. 이벤트 시스템은 이벤트를 전송에 함께 작용하는 일부 컴포넌트로 구성.

- 어떤 게임 오브젝트를 선택할지 관리
- 어떤 입력 모듈을 사용할지 관리
- 레이캐스팅 관리(필요 시)
- 필요에 따라 모든 입력 모듈 업데이트

[https://docs.unity3d.com/kr/2021.3/Manual/EventSystem.html](https://docs.unity3d.com/kr/2021.3/Manual/EventSystem.html)

# Animation & Animator

## Animation

- 특정 행동을 동작으로 표현하는 일련의 변화 과정을 하나의 애니메이션 클립으로 관리한다.

### Mecanim Animation

<img width="803" alt="Untitled 2" src="https://github.com/ssssihoon/CodingTest_Algorithm/assets/127017020/eab24218-65ed-48bd-850c-8cb587f0e474">


## Animator

- 애니메이터(Animator)는 유니티에서 제공하는 애니메이션 시스템 중 하나인 메카님(Mecanim)이라고 불리는 시스템의 핵심 컴포넌트

[https://velog.io/@cedongne/Unity-애니메이션-Animation-Animator-Legacy-Mecanim](https://velog.io/@cedongne/Unity-%EC%95%A0%EB%8B%88%EB%A9%94%EC%9D%B4%EC%85%98-Animation-Animator-Legacy-Mecanim)

# Singleton Pattern

- 싱글톤 패턴은 단일의 인스턴스를 보장하고 이에 대한 전역적인 접근점을 제공하는 패턴
- 위에도 언급했듯이 객체(Object)를 최초 생성한 것을 재사용 하면서 일관성을 보장하게 해준다.
- 자주 호출되는 객체를 new 연산자로 매번 생성하면 메모리 누수의 문제를 방지 할 수 있다.

[https://haedallog.tistory.com/193](https://haedallog.tistory.com/193)

# Coroutine

- 코루틴을 사용하면 작업을 다수의 프레임에 분산할 수 있다.
- Unity에서 코루틴은 실행을 일시 정지하고 제어를 Unity에 반환하지만 중단한 부분에서 다음 프레임을 계속할 수 있는 메서드입니다.

[https://docs.unity3d.com/kr/2022.3/Manual/Coroutines.html](https://docs.unity3d.com/kr/2022.3/Manual/Coroutines.html)
