# Image Process

Classical **computer-vision** image classification implemented as **C# / .NET WinForms desktop applications** (with graphical UIs), rather than Python notebooks. Two complete projects demonstrate hand-crafted feature descriptors feeding a classifier.

## Projects

### `HOG_SVM_OLAN_ARAYÜZLÜ_PROJE/` — HOG + SVM
A WinForms app (`.NET 5`, solution `Teslim.sln`) that classifies images using **Histogram of Oriented Gradients (HOG)** features with an **SVM** classifier.
- UI: `Form1.cs` (main), `FormConfusionMatrix.cs` (results/metrics view)
- Core: `Models/HogDescriptor.cs` (feature extraction), `Models/Data.cs`, `Models/Connet.cs`, custom `InappropriateDimensionException.cs`
- Entry point: `Program.cs`

### `LBP_Olan_Arayüzlü_Proje/` — Local Binary Patterns
A WinForms app (`.NET Core 3.1`, `ImageProcessingProject/`) that classifies images using **Local Binary Patterns (LBP)** texture features.
- UI: `AnaEkran.cs` (main screen), `SplashForm.cs`, `Setting.cs`
- Core: `Models/Data.cs`, `Program.cs`
- Bundled sample images under `Resources/`; the app writes intermediate `grayImages/` and `LBPImages/` at runtime.

## Stack

C# · .NET 5 / .NET Core 3.1 · Windows Forms · HOG · LBP · SVM

## Building & running

Open the `.sln` / project in **Visual Studio** (Windows) and run, or from a terminal:

```powershell
dotnet run --project ".\HOG_SVM_OLAN_ARAYÜZLÜ_PROJE\Teslim\Teslim.csproj"
```

> These are Windows desktop apps and require a Windows + .NET environment. `bin/`, `obj/` and `.vs/` are build artifacts and can be regenerated. See `image-process.doc` for the accompanying report.
