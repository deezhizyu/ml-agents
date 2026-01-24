// This script automatically enables time scale control in ANY scene without manual setup
// Press 1-9 to set time scale, 0 to double current time scale, - to halve it

using UnityEngine;
using UnityEngine.InputSystem;

namespace MLAgentsExamples
{
    /// <summary>
    /// Automatically creates a time scale controller when Unity enters play mode.
    /// No need to manually add this to any scene - it works everywhere!
    /// 
    /// Controls:
    /// - 1-9: Set time scale to that value (1x to 9x)
    /// - 0: Double the current time scale
    /// - Minus (-): Halve the current time scale
    /// </summary>
    public class AutoTimeScaleController : MonoBehaviour
    {
        private static AutoTimeScaleController _instance;
        
        void OnDestroy()
        {
            if (_instance == this)
                _instance = null;
        }
        
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void OnGameStart()
        {
            // Only create one instance
            if (_instance == null)
            {
                var go = new GameObject("TimeScaleController");
                _instance = go.AddComponent<AutoTimeScaleController>();
                DontDestroyOnLoad(go);
                Debug.Log("<color=cyan>[TimeScale] Controller active! Press 1-9 to change time scale, 0 to double, - to halve</color>");
            }
        }

        void Update()
        {
            var keyboard = Keyboard.current;
            if (keyboard == null) return;

            float? newScale = null;
            
            if (keyboard.digit1Key.wasPressedThisFrame) newScale = 1f;
            else if (keyboard.digit2Key.wasPressedThisFrame) newScale = 2f;
            else if (keyboard.digit3Key.wasPressedThisFrame) newScale = 3f;
            else if (keyboard.digit4Key.wasPressedThisFrame) newScale = 4f;
            else if (keyboard.digit5Key.wasPressedThisFrame) newScale = 5f;
            else if (keyboard.digit6Key.wasPressedThisFrame) newScale = 6f;
            else if (keyboard.digit7Key.wasPressedThisFrame) newScale = 7f;
            else if (keyboard.digit8Key.wasPressedThisFrame) newScale = 8f;
            else if (keyboard.digit9Key.wasPressedThisFrame) newScale = 9f;
            else if (keyboard.digit0Key.wasPressedThisFrame)
            {
                Time.timeScale *= 2f;
                Debug.Log($"<color=yellow>[TimeScale] Doubled to {Time.timeScale}x</color>");
            }
            else if (keyboard.minusKey.wasPressedThisFrame)
            {
                Time.timeScale = Mathf.Max(0.5f, Time.timeScale / 2f);
                Debug.Log($"<color=yellow>[TimeScale] Halved to {Time.timeScale}x</color>");
            }

            if (newScale.HasValue)
            {
                Time.timeScale = newScale.Value;
                Debug.Log($"<color=green>[TimeScale] Set to {newScale.Value}x</color>");
            }
        }
    }
}
