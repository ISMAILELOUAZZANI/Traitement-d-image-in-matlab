function MorphologicalOperatorsApp
    % Créer l'interface utilisateur
    fig = uifigure('Name', 'Morphological Operators', 'Position', [100, 100, 600, 400]);

    % Bouton pour charger une image
    btnLoad = uibutton(fig, 'Text', 'Charger l''image', 'Position', [20, 350, 120, 30], ...
        'ButtonPushedFcn', @(btn, event) loadImage());

    % Menu déroulant pour choisir l'opération
    lblOp = uilabel(fig, 'Text', 'Opération:', 'Position', [20, 310, 80, 30]);
    opMenu = uidropdown(fig, 'Items', {'Dilation', 'Erosion', 'Fermeture', 'Ouverture', 'Top Hat Dark', 'Gradient'}, ...
        'Position', [100, 315, 120, 30]);

    % Saisie du seuil
    lblThresh = uilabel(fig, 'Text', 'Seuil:', 'Position', [20, 270, 50, 30]);
    threshInput = uieditfield(fig, 'numeric', 'Value', 127, 'Position', [100, 275, 80, 30]);

    % Bouton pour appliquer l'opération
    btnRun = uibutton(fig, 'Text', 'Run', 'Position', [20, 230, 120, 30], ...
        'ButtonPushedFcn', @(btn, event) applyMorphology());

    % Axes pour afficher les images
    axOriginal = uiaxes(fig, 'Position', [250, 200, 150, 150]);
    title(axOriginal, 'Originale');
    
    axProcessed = uiaxes(fig, 'Position', [420, 200, 150, 150]);
    title(axProcessed, 'Traitée');

    % Variables globales pour stocker l'image
    global img processedImg;

    % Fonction pour charger une image
    function loadImage()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image Files'});
        if isequal(file, 0)
            return;
        end
        img = imread(fullfile(path, file));
        if size(img, 3) == 3
            img = rgb2gray(img); % Convertir en niveaux de gris si nécessaire
        end
        imshow(img, 'Parent', axOriginal);
    end

    % Fonction pour appliquer l'opération morphologique
    function applyMorphology()
        if isempty(img)
            return;
        end
        
        operation = opMenu.Value;
        threshold = threshInput.Value / 255; % Normalisation entre 0 et 1
        binaryImg = img > threshold * 255;  % ✅ Seuillage manuel sans imbinarize

        % Définition du noyau
        kernel = ones(5,5); % ✅ Utilisation d'un simple noyau

        % Sélection de l'opération
        switch operation
            case 'Dilation'
                processedImg = imdilate(binaryImg, kernel);
            case 'Erosion'
                processedImg = imerode(binaryImg, kernel);
            case 'Fermeture'
                processedImg = imdilate(imerode(binaryImg, kernel), kernel);
            case 'Ouverture'
                processedImg = imerode(imdilate(binaryImg, kernel), kernel);
            case 'Top Hat Dark'
                processedImg = binaryImg - imerode(binaryImg, kernel);
            case 'Gradient'
                processedImg = imdilate(binaryImg, kernel) - imerode(binaryImg, kernel); % ✅ Correction Gradient
            otherwise
                processedImg = binaryImg;
        end

        % Affichage de l'image traitée
        imshow(processedImg, [], 'Parent', axProcessed);
    end
end
