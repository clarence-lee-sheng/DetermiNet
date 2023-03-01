// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using System.Threading.Tasks;

// public class Cameras : MonoBehaviour
// {
//     public static ImageSynthesis synth;
//     // Start is called before the first frame update
//     void Start()
//     {
//         // synth = new ImageSynthesis();
//         // synth = Camera.main.GetComponent<ImageSynthesis>();
//         // Debug.Log(Camera.main.GetComponent<ImageSynthesis>());
//         // Debug.Log(synth);
//         // synth.OnSceneChange();
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         // synth.OnSceneChange();
//     }

//     public static void takeSnapshot(string filepath, string filename)
//     {
//         Camera[] allCameras;
//         allCameras = Camera.allCameras;
//         string currTime;
//         currTime = System.DateTime.Now.Millisecond.ToString();
//         ImageSynthesis synth;

//         Debug.Log($"There are {allCameras.Length} cameras");

//         foreach (Camera camera in allCameras)
//         {
//             if (!camera.GetComponent<ImageSynthesis>())
//             {
//                 Debug.Log("No ImageSynthesis component");
//                 continue;
//             }
//             synth = camera.GetComponent<ImageSynthesis>();
//             synth.OnSceneChange();


//             Debug.Log("taking picture with synth");
//             Debug.Log($"{filepath}/{filename}_img.png");
//             synth.Save($"{filename}.png", Constants.imgWidth, Constants.imgHeight, filepath, 2);
//             // change this for different image resolutions 
//             // camera.targetTexture = new RenderTexture(Constants.imgWidth, Constants.imgHeight, 24);
//             // Texture2D snapshot = new Texture2D(Constants.imgWidth, Constants.imgHeight, TextureFormat.RGB24, false);
            
//             // camera.Render(); 
//             // RenderTexture.active = camera.targetTexture;
//             // snapshot.ReadPixels(new Rect(camera.pixelRect),0,0);

//             // byte[] bytes = snapshot.EncodeToPNG();
//             // currTime = System.DateTime.Now.Millisecond.ToString();
//             // System.IO.File.WriteAllBytes($"{filepath}/{filename}_{currTime}.png", bytes);           
//         }
//     }
// }
